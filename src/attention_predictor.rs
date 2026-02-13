use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
use std::{fs, io::Error, io::Read as IoRead};

use crate::grad_accum::AccumAdamW;
use crate::models::RunStr;
use crate::token_utils::STOP_TOKEN;
use crate::{
    attention_block::{AttentionBlock, AttentionBlockConfig},
    token_utils::{tokenize, tokens_to_dict, Dict, DictIndex, GetTokenEmbedding},
};
use candle_core::{DType, Device, Error as CandleError, Tensor, D};
use candle_nn::{self as nn, Module};
use colored::Colorize;
use nn::{VarBuilder, VarMap};

use crate::layer_norm::LayerNorm;

// smoll
const EMBEDDING_SIZE: usize = 128;
const CONTEXT_WINDOW: usize = 64;
const INPUT_SIZE: usize = EMBEDDING_SIZE * CONTEXT_WINDOW;
const NUM_ATTENTION_HEADS: usize = 12;
const HIDDEN_SIZE: usize = 2048;
const NUM_BLOCKS: usize = 2;
pub const CHARS_TO_TRAIN_ON: usize = u64::pow(2, 22) as usize;
pub const FILE_PATH: &str = "common-corpus/level_4/corpus.corpus";
const LR: f64 = 3.0e-4;
const WARMUP_BATCHES: usize = 200;
const EPOCHS: u32 = 4;
const TOKEN_BATCH_SIZE: usize = 256;
pub const TRAINING_SUBSETS: i8 = 3; // we have 12 attention heads - training 4 at a time
const MICRO_BATCH_SIZE: usize = 4;

const NOT_FOUND: &str = "<notfound>";

pub struct Model {
    pub blocks: Vec<AttentionBlock>,
    norm: LayerNorm,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub output_proj: nn::Linear,
    pub embedding: nn::Embedding,
    pub var_map: VarMap,
    pub dict: Dict,
    pub token_index: DictIndex,
    pub index_to_token: Vec<String>,
    pub device: Device,
    pub train_subset_index: i8,
}

fn is_oom_error(e: &CandleError) -> bool {
    if let CandleError::Cuda(inner) = e {
        return inner.to_string().contains("out of memory");
    }
    false
}

const OOM_RETRY_DELAY_SECS: u64 = 30;

impl Model {
    pub fn new(
        dict: Dict,
        var_map: VarMap,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, candle_core::Error> {
        if EMBEDDING_SIZE % NUM_ATTENTION_HEADS != 0 {
            for i in 1..(EMBEDDING_SIZE / 2) {
                if EMBEDDING_SIZE % i == 0 {
                    println!(
                        "Possible num attention heads for this embedding size: {}",
                        i
                    );
                }
            }
            for i in 1..10 {
                println!(
                    "Possible embedding size for this num attention heads: {}",
                    i * NUM_ATTENTION_HEADS
                );
            }

            panic!("The embedding size should be divisible by the number of attention heads!");
        }

        let vocab_size = dict.len();
        let token_index = dict.build_index();
        let index_to_token: Vec<String> = {
            let mut tokens = vec![String::new(); vocab_size];
            for (token, &idx) in &token_index {
                tokens[idx as usize] = token.clone();
            }
            tokens
        };

        let mut blocks = Vec::new();

        for b in 0..NUM_BLOCKS {
            let config: AttentionBlockConfig = AttentionBlockConfig {
                input_size: INPUT_SIZE,
                num_attention_heads: NUM_ATTENTION_HEADS,
                context_window: CONTEXT_WINDOW,
                embedding_size: EMBEDDING_SIZE,
            };

            let block = AttentionBlock::new(config, vb.push_prefix(&format!("block_{}", b)))?;
            blocks.push(block);
        }

        let embedding = nn::embedding(vocab_size, EMBEDDING_SIZE, vb.pp("embedding"))?;
        let norm = LayerNorm::new(EMBEDDING_SIZE, 1e-5, vb.pp("norm"))?;
        let fc1 = nn::linear_b(INPUT_SIZE, HIDDEN_SIZE, true, vb.pp("fc1"))?;
        let fc2 = nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, true, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(HIDDEN_SIZE, EMBEDDING_SIZE, true, vb.pp("fc3"))?;
        let output_proj = nn::linear_b(EMBEDDING_SIZE, vocab_size, true, vb.pp("output_proj"))?;

        println!(
            "Vocab, Embedding Size, Context Window, Epochs, Hidden Size, Num blocks, Num att. heads, LR, Batch Size"
        );
        let output = format!(
            "{}, {}, {}, {}, {}, {}, {}, {}, {}",
            vocab_size,
            EMBEDDING_SIZE,
            CONTEXT_WINDOW,
            EPOCHS,
            HIDDEN_SIZE,
            NUM_BLOCKS,
            NUM_ATTENTION_HEADS,
            LR,
            TOKEN_BATCH_SIZE
        );
        println!("{}", output.on_white().black());

        Ok(Self {
            norm,
            fc1,
            fc2,
            fc3,
            output_proj,
            embedding,
            blocks,
            var_map,
            dict,
            token_index,
            index_to_token,
            device: device.clone(),
            train_subset_index: 0,
        })
    }

    /// input_ids: [batch, CONTEXT_WINDOW] of u32 token indices
    /// returns: [batch, vocab_size] logits
    fn forward(&self, input_ids: &Tensor, train: bool) -> Result<Tensor, candle_core::Error> {
        // Embedding lookup: [batch, CONTEXT_WINDOW] -> [batch, CONTEXT_WINDOW, EMBEDDING_SIZE]
        let embedded = self.embedding.forward(input_ids)?;
        // Flatten: [batch, CONTEXT_WINDOW * EMBEDDING_SIZE] = [batch, INPUT_SIZE]
        let batch_size = embedded.dim(0)?;
        let input = embedded.reshape((batch_size, INPUT_SIZE))?;

        let mut result = input.clone();

        for block in self.blocks.iter() {
            result = block.forward(&result, self.train_subset_index, train)?;
        }

        // Normalize per-token: reshape to [batch, seq, embedding], norm over last dim, flatten back
        let result = result.reshape((batch_size, CONTEXT_WINDOW, EMBEDDING_SIZE))?;
        let result = self.norm.forward(&result)?;
        let result = result.reshape((batch_size, INPUT_SIZE))?;

        let result = self.fc1.forward(&result)?.gelu()?;
        let result = if train {
            nn::ops::dropout(&result, 0.1)?
        } else {
            result
        };
        let result = self.fc2.forward(&result)?.gelu()?;
        let result = if train {
            nn::ops::dropout(&result, 0.1)?
        } else {
            result
        };
        let result = self.fc3.forward(&result)?;
        let result = self.output_proj.forward(&result)?;

        return Ok(result);
    }

    fn token_to_id(&self, token: &str) -> u32 {
        *self
            .token_index
            .get(token)
            .unwrap_or(self.token_index.get(NOT_FOUND).unwrap())
    }

    fn id_to_token(&self, id: u32) -> &str {
        &self.index_to_token[id as usize]
    }

    pub fn run(&self, input_ids: &Vec<u32>, device: &Device) -> Result<String, candle_core::Error> {
        let ids: Vec<u32> = if input_ids.len() > CONTEXT_WINDOW {
            let start = input_ids.len() - CONTEXT_WINDOW;
            input_ids[start..].to_vec()
        } else {
            let pad_id = self.token_to_id(" ");
            let mut padded = vec![pad_id; CONTEXT_WINDOW - input_ids.len()];
            padded.extend_from_slice(input_ids);
            padded
        };

        let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.forward(&input, false)?;

        // Mask <notfound> so it can never be predicted during inference
        let not_found_id = self.token_to_id(NOT_FOUND) as usize;
        let mut logits_vec = logits.to_vec2::<f32>()?[0].clone();
        logits_vec[not_found_id] = f32::NEG_INFINITY;

        // Top-k sampling
        const TOP_K: usize = 5;
        let mut indexed: Vec<(f32, usize)> = logits_vec.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        indexed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let top_k = &indexed[..TOP_K.min(indexed.len())];

        // Softmax over top-k
        let max_logit = top_k[0].0;
        let exps: Vec<f32> = top_k.iter().map(|(v, _)| (v - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let dist = WeightedIndex::new(&probs).unwrap();
        let token_id = top_k[dist.sample(&mut rand::thread_rng())].1 as u32;

        return Ok(self.id_to_token(token_id).to_string());
    }

    pub fn predict_next_token(
        &self,
        input: &str,
        device: &Device,
    ) -> Result<String, candle_core::Error> {
        let tokens = tokenize(&input);
        let input_ids: Vec<u32> = tokens.iter().map(|t| self.token_to_id(t)).collect();
        self.run(&input_ids, device)
    }

    pub fn predict_next_token_greedy(
        &self,
        input: &str,
        device: &Device,
    ) -> Result<String, candle_core::Error> {
        let tokens = tokenize(input);
        let input_ids: Vec<u32> = tokens.iter().map(|t| self.token_to_id(t)).collect();

        let ids: Vec<u32> = if input_ids.len() > CONTEXT_WINDOW {
            let start = input_ids.len() - CONTEXT_WINDOW;
            input_ids[start..].to_vec()
        } else {
            let pad_id = self.token_to_id(" ");
            let mut padded = vec![pad_id; CONTEXT_WINDOW - input_ids.len()];
            padded.extend_from_slice(&input_ids);
            padded
        };

        let input_tensor = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.forward(&input_tensor, false)?;

        let not_found_id = self.token_to_id(NOT_FOUND) as usize;
        let mut logits_vec = logits.to_vec2::<f32>()?[0].clone();
        logits_vec[not_found_id] = f32::NEG_INFINITY;

        let token_id = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap() as u32;

        Ok(self.id_to_token(token_id).to_string())
    }

    pub fn gen_training_data(
        &self,
        tokens_chain: Vec<String>,
        device: &Device,
    ) -> Result<(Tensor, Tensor), candle_core::Error> {
        let mut inputs: Vec<Vec<u32>> = Vec::new();
        let mut targets: Vec<u32> = Vec::new();
        let pad_id = self.token_to_id(" ");

        for index in 1..tokens_chain.len() {
            let target_id = self.token_to_id(&tokens_chain[index]);

            let input_ids: Vec<u32> = if index < CONTEXT_WINDOW {
                let mut padded = vec![pad_id; CONTEXT_WINDOW - index];
                for t in &tokens_chain[0..index] {
                    padded.push(self.token_to_id(t));
                }
                padded
            } else {
                tokens_chain[index - CONTEXT_WINDOW..index]
                    .iter()
                    .map(|t| self.token_to_id(t))
                    .collect()
            };

            inputs.push(input_ids);
            targets.push(target_id);
        }

        let inputs: Vec<Tensor> = inputs
            .iter()
            .map(|ids| Tensor::new(ids.as_slice(), device).unwrap())
            .collect();
        let inputs = Tensor::stack(&inputs, 0)?;
        let targets = Tensor::new(targets.as_slice(), device)?;

        return Ok((inputs, targets));
    }

    pub fn crash_dump(&self, inputs: Tensor, targets: Tensor) -> Result<(), candle_core::Error> {
        self.save_to_path("data/model");
        self.print_stats()?;

        let batch_size = inputs.dim(0)?;
        println!("Batch size {}", batch_size);

        let input_ids = inputs.to_vec2::<u32>()?;
        let target_ids = targets.to_vec1::<u32>()?;

        for i in 0..batch_size {
            let input_tokens: Vec<&str> = input_ids[i]
                .iter()
                .map(|&id| self.id_to_token(id))
                .collect();
            println!(
                "batch[{}] {:?} -> {:?}",
                i,
                input_tokens,
                self.id_to_token(target_ids[i])
            );
        }

        return Ok(());
    }

    pub fn simple_train(
        &mut self,
        tokens_chain: Vec<String>,
        device: &Device,
    ) -> Result<(), candle_core::Error> {
        let start_time = std::time::Instant::now();
        let epochs: u32 = EPOCHS;

        let corpus_level_pre = FILE_PATH
            .split('/')
            .find_map(|s| s.strip_prefix("level_").and_then(|n| n.parse::<u32>().ok()))
            .unwrap_or(0);
        println!(
            "Corpus_Level\tDict_Size\tEmbedding_Size\tContext_Window\tEpochs\tHidden_Size\tNum_blocks\tNum_att_heads\tLR\tBatch_Size"
        );
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            corpus_level_pre, self.dict.len(), EMBEDDING_SIZE, CONTEXT_WINDOW,
            EPOCHS, HIDDEN_SIZE, NUM_BLOCKS, NUM_ATTENTION_HEADS, LR, TOKEN_BATCH_SIZE
        );

        let mut optimizer = AccumAdamW::new(self.var_map.all_vars(), LR)?;

        // Pre-generate all (input, target) pairs from the full corpus. Each sample is a
        // self-contained context window, so shuffling them across epochs is safe — it doesn't
        // break ordering within any article.
        let (all_inputs, all_targets) = self.gen_training_data(tokens_chain, device)?;
        let num_samples = all_inputs.dim(0)?;
        let _num_batches = (num_samples + MICRO_BATCH_SIZE - 1) / MICRO_BATCH_SIZE;

        let mut rng = rand::thread_rng();
        let mut global_step: usize = 0;

        for epoch in 0..epochs {
            let mut loss_stat: f32 = 1.0;

            // Shuffle sample indices each epoch so batches draw from across the corpus
            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut rng);

            let batch_count = (num_samples + TOKEN_BATCH_SIZE - 1) / TOKEN_BATCH_SIZE;
            for j in 0..batch_count {
                let batch_start = j * TOKEN_BATCH_SIZE;
                let batch_end = (batch_start + TOKEN_BATCH_SIZE).min(num_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // Linear warmup: ramp LR from 0 to LR over the first WARMUP_BATCHES steps
                let warmup_lr = LR * (global_step as f64 / WARMUP_BATCHES as f64).min(1.0);
                optimizer.set_learning_rate(warmup_lr);
                global_step += 1;

                // Rotate train subset once per batch
                self.train_subset_index = (self.train_subset_index + 1) % TRAINING_SUBSETS;

                loop {
                    let result: Result<(), CandleError> = (|| {
                        // Build this batch's tensors from shuffled indices
                        let batch_inputs: Vec<Tensor> = batch_indices
                            .iter()
                            .map(|&i| all_inputs.narrow(0, i, 1).unwrap().squeeze(0).unwrap())
                            .collect();
                        let batch_targets: Vec<u32> = batch_indices
                            .iter()
                            .map(|&i| {
                                all_targets
                                    .narrow(0, i, 1)
                                    .unwrap()
                                    .to_vec1::<u32>()
                                    .unwrap()[0]
                            })
                            .collect();
                        let inputs = Tensor::stack(&batch_inputs, 0)?;
                        let targets = Tensor::new(batch_targets.as_slice(), device)?;

                        // Split into micro-batches for gradient accumulation
                        let num_samples = inputs.dim(0)?;
                        let num_micro = (num_samples + MICRO_BATCH_SIZE - 1) / MICRO_BATCH_SIZE;

                        for m in 0..num_micro {
                            let micro_start = m * MICRO_BATCH_SIZE;
                            let micro_len = MICRO_BATCH_SIZE.min(num_samples - micro_start);
                            let micro_inputs = inputs.narrow(0, micro_start, micro_len)?;
                            let micro_targets = targets.narrow(0, micro_start, micro_len)?;

                            let predictions = self.forward(&micro_inputs, true)?;

                            let loss = nn::loss::cross_entropy(&predictions, &micro_targets)?;
                            loss_stat = loss.to_vec0::<f32>()?;

                            if loss_stat.is_nan() {
                                self.crash_dump(inputs.clone(), targets.clone())?;
                                panic!("Loss is nan, gradient probably exploded or vanished.");
                            }

                            optimizer.accumulate(&loss)?;
                        }

                        optimizer.step()?;
                        Ok(())
                    })();

                    match result {
                        Ok(()) => break,
                        Err(e) if is_oom_error(&e) => {
                            optimizer.clear_accumulated();
                            eprintln!(
                                "\nCUDA OOM on batch {j}, retrying in {OOM_RETRY_DELAY_SECS}s...",
                            );
                            std::thread::sleep(std::time::Duration::from_secs(
                                OOM_RETRY_DELAY_SECS,
                            ));
                        }
                        Err(e) => return Err(e),
                    }
                }

                if j % 200 == 0 {
                    println!(
                        "\rEpoch {:4}/{:4} Batch {:4}/{:4} Loss = {:.6}",
                        epoch, epochs, j, batch_count, loss_stat
                    );
                    let prediction = self.run_str("Two birds", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    print!("The birds|>{:.40}", prediction);
                    let prediction = self.run_str("The cat", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    print!(" The cat|>{:.40}", prediction);
                    let prediction = self.run_str("The dog", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    println!(" The dog|>{:.40}", prediction);
                    let prediction = self.run_str("The fish", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    print!("The fish|>{:.40}", prediction);
                    let prediction = self.run_str("A sailboat", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    print!(" A sailboat|>{:.40}", prediction);
                    let prediction = self.run_str("A carrot", 15)?;
                    let prediction = prediction.replace("\n", "_");
                    println!(" A carrot|>{:.40}", prediction);
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }

            println!(
                "\rEpoch {:6}/{:6} : Loss = {:.6}              ",
                epoch, epochs, loss_stat
            );

            if epoch % 40 == 0 {
                self.save_to_path("data/model");
            }
        }

        let elapsed = start_time.elapsed();
        let h = elapsed.as_secs() / 3600;
        let m = (elapsed.as_secs() % 3600) / 60;
        let s = elapsed.as_secs() % 60;
        let time_str = format!("{}:{:02}:{:02}", h, m, s);

        let date = std::process::Command::new("date")
            .arg("+%d/%m/%Y")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let git_hash = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        let corpus_level = FILE_PATH
            .split('/')
            .find_map(|s| s.strip_prefix("level_").and_then(|n| n.parse::<u32>().ok()))
            .unwrap_or(0);

        let entry = serde_json::json!({
            "Corpus_Level": corpus_level,
            "Dict_Size": self.dict.len(),
            "Embedding_Size": EMBEDDING_SIZE,
            "Context_Window": CONTEXT_WINDOW,
            "Epochs": EPOCHS,
            "Hidden_Size": HIDDEN_SIZE,
            "Num_blocks": NUM_BLOCKS,
            "Num_att_heads": NUM_ATTENTION_HEADS,
            "LR": LR,
            "Batch_Size": TOKEN_BATCH_SIZE,
            "State_of_the_code": git_hash,
            "Time_to_train": time_str,
            "Date": date,
        });

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("training_log.json")
        {
            use std::io::Write as IoWrite2;
            let _ = writeln!(file, "{}", serde_json::to_string(&entry).unwrap());
        }

        Ok(())
    }

    pub fn print_stats(&self) -> Result<(), candle_core::Error> {
        println!("Model stats:");
        println!("Dict size: {}", self.dict.len());

        // print min, max, mean, std of all tensors
        for var in self.var_map.all_vars().iter() {
            let min = var.min_all()?.to_vec0::<f32>()?;
            let max = var.max_all()?.to_vec0::<f32>()?;
            let mean = var.mean_all()?.to_vec0::<f32>()?;
            let variance = var.flatten_all()?.var(D::Minus1)?.to_vec0::<f32>()?;
            println!(
                "{}: min: {:.3}, max: {:.3}, mean: {:.3}, std: {:.3}",
                "", min, max, mean, variance
            );
        }

        Ok(())
    }

    pub fn save_to_path(&self, path: &str) {
        let var_map_path = format!("{}.safetensors", path);
        self.var_map.save(var_map_path.as_str()).unwrap();

        let dict_words = self
            .dict
            .iter()
            .map(|(word, _)| word.clone())
            .collect::<Vec<String>>();

        let dict_path = format!("{}.dict", path);
        let file = fs::File::create(dict_path).unwrap();
        serde_json::to_writer(file, &dict_words).unwrap();
    }

    pub fn load_from_path(path: &str, device: &Device) -> Result<Self, Error> {
        let dict_path = format!("{}.dict", path);
        let file = fs::File::open(&dict_path).unwrap();
        let dict_words: Vec<String> = serde_json::from_reader(file).unwrap();
        let dict = tokens_to_dict(dict_words);

        let mut model = create_model(&dict, device).unwrap();

        let var_map_path = format!("{}.safetensors", path);
        model.var_map.load(var_map_path.as_str()).unwrap();

        Ok(model)
    }
}

impl RunStr for Model {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error> {
        let mut output = String::new();
        let tokens = tokenize(input);
        let mut input_ids: Vec<u32> = tokens.iter().map(|t| self.token_to_id(t)).collect();

        for _ in 0..len {
            let prediction = self.run(&input_ids, &self.device)?;
            let pred_id = self.token_to_id(&prediction);
            input_ids.push(pred_id);

            if prediction == STOP_TOKEN {
                break;
            }

            output.push_str(&prediction);
        }

        return Ok(output);
    }
}

pub fn create_model(dict: &Dict, device: &Device) -> Result<Model, candle_core::Error> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Model::new(dict.clone(), varmap, vb, device)?;

    Ok(model)
}

pub fn get_device() -> Result<Device, candle_core::Error> {
    if cfg!(target_os = "macos") {
        let device = Device::new_metal(0)?;
        match &device {
            Device::Metal(m) => m,
            _ => panic!("Device is not Metal"),
        };
        return Ok(device);
    } else {
        return Device::new_cuda(0);
    }
}

fn read_n_chars(file_path: &str, n: u64) -> Result<String, std::io::Error> {
    let file = fs::File::open(file_path)?;
    let mut content = String::new();
    let mut handle = file.take(n);
    handle.read_to_string(&mut content)?;
    Ok(content)
}

pub fn get_pretrained_dict(file_path: &str) -> Result<(Dict, Vec<String>), candle_core::Error> {
    let char_count = CHARS_TO_TRAIN_ON as u64;

    println!("Reading {} chars from file: {}", char_count, file_path);
    let content = read_n_chars(file_path, char_count)?;
    println!("Read {} chars", content.len());
    let tokens: Vec<String> = tokenize(&content).to_vec();
    let dict = tokens_to_dict(tokens.clone());
    println!("Dict size: {}", dict.len());

    let lorem_tokens = tokenize("lorem ipsum et dolor sit amet");
    let hello_world_tokens = tokenize("hello world");
    let sys_tokens = vec![String::from(NOT_FOUND), String::from(STOP_TOKEN)];

    let tokens = [tokens, lorem_tokens, hello_world_tokens, sys_tokens].concat();

    let dict = tokens_to_dict(tokens.clone());

    return Ok((dict, tokens));
}
