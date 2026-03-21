use rand::distributions::{Alphanumeric, WeightedIndex};
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use std::{fs, io::Error};

use crate::grad_accum::AdamW;
use crate::layer_norm::LayerNorm;
use crate::model_tests::{per_epoch_scores, print_epoch_stats_header};
use crate::models::{PredictGreedy, RunStr};
use crate::token_utils::STOP_TOKEN;
use crate::train_config::TrainConfig;
use crate::{
    attention_block::{AttentionBlock, AttentionBlockConfig},
    token_utils::{tokens_to_dict, Bpe, Dict, DictIndex, GetTokenEmbedding, NUM_BPE_MERGES},
};
use candle_core::{DType, Device, Error as CandleError, Tensor, D};
use candle_nn::{self as nn, Module};
use colored::Colorize;
use nn::{VarBuilder, VarMap};

const NOT_FOUND: &str = "<notfound>";

pub struct Model {
    pub blocks: Vec<AttentionBlock>,
    pub embedding: nn::Embedding,
    pre_proj_norm: LayerNorm,
    pre_proj_in: nn::Linear,
    pre_proj_out: nn::Linear,
    pub var_map: VarMap,
    pub dict: Dict,
    pub token_index: DictIndex,
    pub index_to_token: Vec<String>,
    pub device: Device,
    pub model_id: String,
    pub bpe: Bpe,
    pub config: TrainConfig,
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
        bpe: Bpe,
        var_map: VarMap,
        vb: VarBuilder,
        device: &Device,
        config: TrainConfig,
    ) -> Result<Self, candle_core::Error> {
        if config.embedding_size % config.num_attention_heads != 0 {
            for i in 1..(config.embedding_size / 2) {
                if config.embedding_size % i == 0 {
                    println!(
                        "Possible num attention heads for this embedding size: {}",
                        i
                    );
                }
            }
            for i in 1..10 {
                println!(
                    "Possible embedding size for this num attention heads: {}",
                    i * config.num_attention_heads
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

        for b in 0..config.num_blocks {
            let block_config = AttentionBlockConfig {
                num_attention_heads: config.num_attention_heads,
                context_window: config.context_window,
                embedding_size: config.embedding_size,
                ffn_hidden: config.ffn_hidden,
            };

            let block = AttentionBlock::new(block_config, vb.push_prefix(&format!("block_{}", b)))?;
            blocks.push(block);
        }

        let embedding = nn::embedding(vocab_size, config.embedding_size, vb.pp("embedding"))?;
        let pre_proj_norm = LayerNorm::new(config.embedding_size, 1e-5, vb.pp("pre_proj_norm"))?;
        let pre_proj_in = nn::linear_b(config.embedding_size, config.ffn_hidden, true, vb.pp("pre_proj_in"))?;
        let pre_proj_out = nn::linear_b(config.ffn_hidden, config.embedding_size, true, vb.pp("pre_proj_out"))?;

        println!(
            "Vocab, Embedding Size, Context Window, Epochs, Hidden Size, Num blocks, Num att. heads, LR, Batch Size"
        );
        let output = format!(
            "{}, {}, {}, {}, {}, {}, {}, {}, {}",
            vocab_size,
            config.embedding_size,
            config.context_window,
            config.epochs,
            config.ffn_hidden,
            config.num_blocks,
            config.num_attention_heads,
            config.lr,
            config.token_batch_size
        );
        println!("{}", output.on_white().black());

        let model_id: String = rand::thread_rng()
            .sample_iter(Alphanumeric)
            .take(12)
            .map(char::from)
            .collect();

        Ok(Self {
            embedding,
            pre_proj_norm,
            pre_proj_in,
            pre_proj_out,
            blocks,
            var_map,
            dict,
            token_index,
            index_to_token,
            device: device.clone(),
            model_id,
            bpe,
            config,
        })
    }

    /// input_ids: [batch, CONTEXT_WINDOW] of u32 token indices
    /// returns: [batch, vocab_size] logits
    fn forward(&self, input_ids: &Tensor, train: bool) -> Result<Tensor, candle_core::Error> {
        // Embedding lookup: [batch, CONTEXT_WINDOW] -> [batch, CONTEXT_WINDOW, EMBEDDING_SIZE]
        let embedded = self.embedding.forward(input_ids)?;
        // Add positional encoding once before the attention blocks
        let embedded = embedded.broadcast_add(self.blocks[0].position_encoding())?;

        let mut result = embedded;

        for block in self.blocks.iter() {
            result = block.forward(&result, train)?;
        }

        // Take last token's representation: [batch, emb]
        let result = result
            .narrow(1, self.config.context_window - 1, 1)?
            .squeeze(1)?
            .contiguous()?;

        // Intermediate projection to decouple reasoning space from embedding space
        let pre_proj_residual = result.clone();
        let result = self.pre_proj_norm.forward(&result)?;
        let result = self.pre_proj_in.forward(&result)?.gelu()?;
        let result = (self.pre_proj_out.forward(&result)? + pre_proj_residual)?;

        // Weight-tied output projection: [batch, emb] @ [emb, vocab] -> [batch, vocab]
        let result = result.matmul(&self.embedding.embeddings().t()?)?;

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
        let cw = self.config.context_window;
        let ids: Vec<u32> = if input_ids.len() > cw {
            let start = input_ids.len() - cw;
            input_ids[start..].to_vec()
        } else {
            let pad_id = self.token_to_id(" ");
            let mut padded = vec![pad_id; cw - input_ids.len()];
            padded.extend_from_slice(input_ids);
            padded
        };

        let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.forward(&input, false)?;

        // Mask <notfound> so it can never be predicted during inference
        let not_found_id = self.token_to_id(NOT_FOUND) as usize;
        let mut logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();
        logits_vec[not_found_id] = f32::NEG_INFINITY;

        // Top-k sampling
        const TOP_K: usize = 5;
        let mut indexed: Vec<(f32, usize)> = logits_vec
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
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
        let tokens = self.bpe.tokenize(input);
        let input_ids: Vec<u32> = tokens.iter().map(|t| self.token_to_id(t)).collect();
        self.run(&input_ids, device)
    }

    pub fn predict_next_token_greedy(
        &self,
        input: &str,
        device: &Device,
    ) -> Result<String, candle_core::Error> {
        let tokens = self.bpe.tokenize(input);
        let input_ids: Vec<u32> = tokens.iter().map(|t| self.token_to_id(t)).collect();

        let cw = self.config.context_window;
        let ids: Vec<u32> = if input_ids.len() > cw {
            let start = input_ids.len() - cw;
            input_ids[start..].to_vec()
        } else {
            let pad_id = self.token_to_id(" ");
            let mut padded = vec![pad_id; cw - input_ids.len()];
            padded.extend_from_slice(&input_ids);
            padded
        };

        let input_tensor = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let logits = self.forward(&input_tensor, false)?;

        let not_found_id = self.token_to_id(NOT_FOUND) as usize;
        let mut logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();
        logits_vec[not_found_id] = f32::NEG_INFINITY;

        let token_id = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap() as u32;

        Ok(self.id_to_token(token_id).to_string())
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
        base_lr: f64,
    ) -> Result<(), candle_core::Error> {
        let start_time = std::time::Instant::now();
        let epochs = self.config.epochs;
        let context_window = self.config.context_window;
        let token_batch_size = self.config.token_batch_size;
        let warmup_batches = self.config.warmup_batches;

        let corpus_level = self.config.file_path
            .split('/')
            .find_map(|s| s.strip_prefix("level_").and_then(|n| n.parse::<u32>().ok()))
            .unwrap_or(0);

        let git_hash = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();

        print_epoch_stats_header();

        println!(
            "Corpus_Level\tDict_Size\tEmbedding_Size\tContext_Window\tEpochs\tHidden_Size\tNum_blocks\tNum_att_heads\tLR\tBatch_Size"
        );
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            corpus_level,
            self.dict.len(),
            self.config.embedding_size,
            context_window,
            epochs,
            self.config.ffn_hidden,
            self.config.num_blocks,
            self.config.num_attention_heads,
            base_lr,
            token_batch_size
        );

        let mut optimizer = AdamW::new(self.var_map.all_vars(), base_lr)?;

        let pad_id = self.token_to_id(" ");
        let token_ids: Vec<u32> = tokens_chain.iter().map(|t| self.token_to_id(t)).collect();
        let num_samples = token_ids.len().saturating_sub(1);

        const CHUNK_SIZE: usize = 100_000;
        let seqs_per_batch = (token_batch_size / context_window).max(1);

        let mut rng = rand::thread_rng();
        let mut global_step: usize = 0;
        let batch_count = (num_samples + seqs_per_batch - 1) / seqs_per_batch;
        let total_steps = epochs as usize * batch_count;
        let lr_min = base_lr * 0.1;

        for epoch in 0..epochs {
            let mut loss_stat: f32 = 1.0;
            let mut last_lr = base_lr;

            // Shuffle sample indices each epoch
            let mut indices: Vec<u32> = (0..num_samples as u32).collect();
            indices.shuffle(&mut rng);
            let mut batch_timer = std::time::Instant::now();
            let mut j = 0usize;

            for chunk_start in (0..num_samples).step_by(CHUNK_SIZE) {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(num_samples);
                let chunk_indices = &indices[chunk_start..chunk_end];
                let chunk_len = chunk_indices.len();

                // Build this chunk on CPU (~50 MB) then upload to GPU
                let mut flat_seqs: Vec<u32> = Vec::with_capacity(chunk_len * context_window);
                let mut flat_tgts: Vec<u32> = Vec::with_capacity(chunk_len);
                for &idx in chunk_indices {
                    let idx = idx as usize;
                    let target = token_ids[idx + 1];
                    let start = idx.saturating_sub(context_window - 1);
                    let window = &token_ids[start..=idx];
                    let pad_len = context_window - window.len();
                    flat_seqs.extend(std::iter::repeat(pad_id).take(pad_len));
                    flat_seqs.extend_from_slice(window);
                    flat_tgts.push(target);
                }
                let chunk_seqs = Tensor::from_vec(flat_seqs, (chunk_len, context_window), device)?;
                let chunk_tgts = Tensor::from_vec(flat_tgts, chunk_len, device)?;

                let chunk_batch_count = (chunk_len + seqs_per_batch - 1) / seqs_per_batch;
                for ci in 0..chunk_batch_count {
                    let bs = ci * seqs_per_batch;
                    let be = (bs + seqs_per_batch).min(chunk_len);
                    let all_inputs = chunk_seqs.narrow(0, bs, be - bs)?;
                    let all_targets = chunk_tgts.narrow(0, bs, be - bs)?;

                // Constant LR, or linear warmup then cosine decay
                let lr = if self.config.no_warmup {
                    base_lr
                } else if global_step < warmup_batches {
                    base_lr * ((global_step + 1) as f64 / warmup_batches as f64)
                } else {
                    let decay_steps = (total_steps - warmup_batches).max(1);
                    let progress = (global_step - warmup_batches) as f64 / decay_steps as f64;
                    lr_min
                        + 0.5 * (base_lr - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
                };
                optimizer.set_learning_rate(lr);
                last_lr = lr;
                global_step += 1;

                let mut oom_retries = 0u32;
                loop {
                    let result: Result<(), CandleError> = (|| {
                        let predictions = self.forward(&all_inputs, true)?;
                        let loss = nn::loss::cross_entropy(&predictions.to_dtype(DType::F32)?, &all_targets)?;

                        // Only sync GPU→CPU every 200 batches to avoid stalling the pipeline
                        if j % 200 == 0 {
                            loss_stat = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;
                            if loss_stat.is_nan() {
                                self.crash_dump(all_inputs.clone(), all_targets.clone())?;
                                panic!("Loss is nan, gradient probably exploded or vanished.");
                            }
                        }

                        optimizer.step(&loss)?;
                        Ok(())
                    })();

                    match result {
                        Ok(()) => break,
                        Err(e) if is_oom_error(&e) => {
                            oom_retries += 1;
                            if oom_retries >= 3 {
                                eprintln!("\nCUDA OOM on batch {j}: too many retries, aborting.");
                                return Err(e);
                            }
                            eprintln!(
                                "\nCUDA OOM on batch {j} (retry {oom_retries}/3), retrying in {OOM_RETRY_DELAY_SECS}s...",
                            );
                            std::thread::sleep(std::time::Duration::from_secs(
                                OOM_RETRY_DELAY_SECS,
                            ));
                        }
                        Err(e) => return Err(e),
                    }
                }

                if j % 200 == 0 {
                    let elapsed = batch_timer.elapsed();
                    batch_timer = std::time::Instant::now();
                    let ms_per_batch = if j > 0 {
                        elapsed.as_secs_f64() * 1000.0 / 200.0
                    } else {
                        0.0
                    };
                    let batches_done = epoch as usize * batch_count + j;
                    let batches_left = total_steps - batches_done;
                    let eta_secs = batches_left as f64 * ms_per_batch / 1000.0;
                    let eta_str = if ms_per_batch > 0.0 {
                        let h = (eta_secs / 3600.0) as u64;
                        let m = ((eta_secs % 3600.0) / 60.0) as u64;
                        format!("{}h{}m left", h, m)
                    } else {
                        "?".to_string()
                    };
                    println!(
                        "\rEpoch {:4}/{:4} Batch {:4}/{:4} Loss = {:.6} LR = {:.2e} ({:.0}ms/batch, {})",
                        epoch, epochs, j, batch_count, loss_stat, lr, ms_per_batch, eta_str
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
                    j += 1;
                } // end ci loop
            } // end chunk loop

            println!(
                "\rEpoch {:6}/{:6} : Loss = {:.6}              ",
                epoch, epochs, loss_stat
            );

            self.save_to_path("data/model");
            println!("Saved model checkpoint.");

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

            match per_epoch_scores(self, device) {
                Ok((score_l2, score_qa, score_json)) => {
                    let entry = serde_json::json!({
                        "Epoch": epoch,
                        "Model_ID": self.model_id,
                        "Corpus_Level": corpus_level,
                        "Dict_Size": self.dict.len(),
                        "Embedding_Size": self.config.embedding_size,
                        "Context_Window": context_window,
                        "Epochs": epochs,
                        "Hidden_Size": self.config.ffn_hidden,
                        "Num_blocks": self.config.num_blocks,
                        "Num_att_heads": self.config.num_attention_heads,
                        "LR": last_lr,
                        "Batch_Size": token_batch_size,
                        "State_of_the_code": git_hash,
                        "Time_to_train": time_str,
                        "Self_Test_Score_L2": score_l2,
                        "QA_Test_Score": score_qa,
                        "JSON_Test_Score": score_json,
                        "Date": date,
                    });
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("per_epoch_stats.log")
                    {
                        use std::io::Write as IoWrite3;
                        let _ = writeln!(file, "{}", serde_json::to_string(&entry).unwrap());
                    }
                    println!(
                        "Epoch {} scores: L2={:.3} QA={:.3} JSON={:.3} LR={:.2e}",
                        epoch, score_l2, score_qa, score_json, last_lr
                    );
                }
                Err(e) => eprintln!("Epoch {} test failed: {}", epoch, e),
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

        let entry = serde_json::json!({
            "Model_ID": self.model_id,
            "Corpus_Level": corpus_level,
            "Dict_Size": self.dict.len(),
            "Embedding_Size": self.config.embedding_size,
            "Context_Window": context_window,
            "Epochs": epochs,
            "Hidden_Size": self.config.ffn_hidden,
            "Num_blocks": self.config.num_blocks,
            "Num_att_heads": self.config.num_attention_heads,
            "LR": base_lr,
            "Batch_Size": token_batch_size,
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

    /// LR range test (Leslie Smith 2018).
    ///
    /// Ramps LR log-linearly from `lr_lo` to `lr_hi` over `max_batches` steps.
    /// Tracks EMA-smoothed loss (beta=0.98, bias-corrected).  Stops early when
    /// smoothed loss exceeds 4× the minimum seen (model diverging).
    /// Prints a TSV table and writes the same rows to `lr_range_test.tsv`.
    pub fn lr_range_test(
        &mut self,
        tokens_chain: Vec<String>,
        device: &Device,
        lr_lo: f64,
        lr_hi: f64,
        max_batches: usize,
    ) -> Result<(), candle_core::Error> {
        use std::io::Write as IoWrite;

        let context_window = self.config.context_window;
        let token_batch_size = self.config.token_batch_size;
        let seqs_per_batch = (token_batch_size / context_window).max(1);

        let pad_id = self.token_to_id(" ");
        let token_ids: Vec<u32> = tokens_chain.iter().map(|t| self.token_to_id(t)).collect();
        let num_samples = token_ids.len().saturating_sub(1);

        let mut optimizer = AdamW::new(self.var_map.all_vars(), lr_lo)?;

        // Shuffle once
        let mut rng = rand::thread_rng();
        let mut indices: Vec<u32> = (0..num_samples as u32).collect();
        indices.shuffle(&mut rng);

        const CHUNK_SIZE: usize = 100_000;
        const BETA: f64 = 0.98;

        let mut out_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("lr_range_test.tsv")
            .ok();

        let header = "step\tlr\tsmoothed_loss\traw_loss";
        println!("{}", header);
        if let Some(f) = &mut out_file {
            let _ = writeln!(f, "{}", header);
        }

        let mut step = 0usize;
        let mut ema: f64 = 0.0;
        let mut min_smooth = f64::MAX;
        // Collect (lr, smoothed_loss) for post-run analysis
        let mut history: Vec<(f64, f64)> = Vec::with_capacity(max_batches);

        'outer: for chunk_start in (0..num_samples).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(num_samples);
            let chunk_indices = &indices[chunk_start..chunk_end];
            let chunk_len = chunk_indices.len();

            let mut flat_seqs: Vec<u32> = Vec::with_capacity(chunk_len * context_window);
            let mut flat_tgts: Vec<u32> = Vec::with_capacity(chunk_len);
            for &idx in chunk_indices {
                let idx = idx as usize;
                let target = token_ids[idx + 1];
                let start = idx.saturating_sub(context_window - 1);
                let window = &token_ids[start..=idx];
                let pad_len = context_window - window.len();
                flat_seqs.extend(std::iter::repeat(pad_id).take(pad_len));
                flat_seqs.extend_from_slice(window);
                flat_tgts.push(target);
            }
            let chunk_seqs = Tensor::from_vec(flat_seqs, (chunk_len, context_window), device)?;
            let chunk_tgts = Tensor::from_vec(flat_tgts, chunk_len, device)?;

            let chunk_batch_count = (chunk_len + seqs_per_batch - 1) / seqs_per_batch;
            for ci in 0..chunk_batch_count {
                if step >= max_batches {
                    break 'outer;
                }

                // Log-linear LR ramp
                let t = step as f64 / (max_batches - 1).max(1) as f64;
                let lr = lr_lo * (lr_hi / lr_lo).powf(t);
                optimizer.set_learning_rate(lr);

                let bs = ci * seqs_per_batch;
                let be = (bs + seqs_per_batch).min(chunk_len);
                let inputs = chunk_seqs.narrow(0, bs, be - bs)?;
                let targets = chunk_tgts.narrow(0, bs, be - bs)?;

                let predictions = self.forward(&inputs, true)?;
                let loss = nn::loss::cross_entropy(&predictions.to_dtype(DType::F32)?, &targets)?;
                optimizer.step(&loss)?;

                let raw: f64 = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;

                // EMA with bias correction
                ema = BETA * ema + (1.0 - BETA) * raw;
                let smooth = ema / (1.0 - BETA.powi((step + 1) as i32));

                if smooth < min_smooth {
                    min_smooth = smooth;
                }
                history.push((lr, smooth));

                let row = format!("{}\t{:.4e}\t{:.6}\t{:.6}", step, lr, smooth, raw);
                println!("{}", row);
                if let Some(f) = &mut out_file {
                    let _ = writeln!(f, "{}", row);
                }

                // Early stop: diverging
                if smooth > 4.0 * min_smooth {
                    println!("# Early stop: loss diverged (smooth={:.4} > 4×min={:.4})", smooth, min_smooth);
                    break 'outer;
                }

                step += 1;
            }
        }

        // Find the steepest descent window (most negative slope over a 10% window).
        // This is the LR where loss is dropping fastest — the recommended training LR.
        let best_lr = if history.len() >= 10 {
            let window = (history.len() / 10).max(5);
            let (best_idx, _) = history
                .windows(window)
                .enumerate()
                .map(|(i, w)| {
                    let slope = (w.last().unwrap().1 - w.first().unwrap().1) / window as f64;
                    (i, slope)
                })
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            // Use the LR at the center of the steepest window
            let center = best_idx + window / 2;
            Some(history[center].0)
        } else {
            None
        };

        println!("# Results written to lr_range_test.tsv");
        match best_lr {
            Some(lr) => println!("# Recommended LR: {:.2e}  (steepest loss descent)", lr),
            None => println!("# Not enough data to recommend LR"),
        }
        Ok(())
    }

    pub fn count_params(&self) -> usize {
        let data = self.var_map.data().lock().unwrap();
        let mut total = 0usize;
        let mut entries: Vec<(&String, usize)> = data
            .iter()
            .map(|(name, var)| (name, var.elem_count()))
            .collect();
        entries.sort_by_key(|(name, _)| name.as_str());
        for (name, count) in &entries {
            println!("  {:60} {:>10} params", name, count);
            total += count;
        }
        let embedding_params = self.dict.len() * self.config.embedding_size;
        println!(
            "  {:60} {:>10} params  (weight-tied, counted above)",
            "embedding (output projection)", embedding_params
        );
        println!("  {:60} {:>10}", "TOTAL", total);
        println!(
            "  {:60} {:>10}  (excl. embedding)",
            "TOTAL non-embedding",
            total - embedding_params
        );
        total
    }

    pub fn print_stats(&self) -> Result<(), candle_core::Error> {
        println!("Model stats:");
        println!("Dict size: {}", self.dict.len());

        // print min, max, mean, std of all tensors
        for var in self.var_map.all_vars().iter() {
            let min = var.min_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()?;
            let max = var.max_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()?;
            let mean = var.mean_all()?.to_dtype(DType::F32)?.to_vec0::<f32>()?;
            let variance = var.flatten_all()?.to_dtype(DType::F32)?.var(D::Minus1)?.to_vec0::<f32>()?;
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

        let id_path = format!("{}.id", path);
        fs::write(id_path, &self.model_id).unwrap();

        let bpe_path = format!("{}.bpe", path);
        self.bpe.save(&bpe_path).unwrap();

        let config_path = format!("{}.config.json", path);
        if let Ok(file) = fs::File::create(&config_path) {
            let _ = serde_json::to_writer_pretty(file, &self.config);
        }
    }

    pub fn load_from_path(path: &str, device: &Device) -> Result<Self, Error> {
        let dict_path = format!("{}.dict", path);
        let file = fs::File::open(&dict_path).unwrap();
        let dict_words: Vec<String> = serde_json::from_reader(file).unwrap();
        let dict = tokens_to_dict(dict_words);

        let bpe_path = format!("{}.bpe", path);
        let bpe = Bpe::load(&bpe_path).unwrap_or_else(|_| Bpe::new_empty());

        // Load saved config if present, otherwise fall back to env vars
        let config = {
            let config_path = format!("{}.config.json", path);
            fs::File::open(&config_path)
                .ok()
                .and_then(|f| serde_json::from_reader(f).ok())
                .unwrap_or_else(TrainConfig::from_env)
        };

        let mut model = create_model(&dict, bpe, device, config).unwrap();

        let var_map_path = format!("{}.safetensors", path);
        model.var_map.load(var_map_path.as_str()).unwrap();

        let id_path = format!("{}.id", path);
        if let Ok(id) = fs::read_to_string(&id_path) {
            model.model_id = id.trim().to_string();
        }

        Ok(model)
    }
}

impl PredictGreedy for Model {
    fn predict_next_token_greedy(
        &self,
        input: &str,
        device: &Device,
    ) -> Result<String, candle_core::Error> {
        self.predict_next_token_greedy(input, device)
    }
}

impl RunStr for Model {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error> {
        let mut output = String::new();
        let tokens = self.bpe.tokenize(input);
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

fn gpu_compute_cap_x10() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let cap: f32 = s.trim().parse().ok()?;
    Some((cap * 10.0).round() as u32)
}

pub fn create_model(dict: &Dict, bpe: Bpe, device: &Device, mut config: TrainConfig) -> Result<Model, candle_core::Error> {
    if config.use_bf16 {
        match gpu_compute_cap_x10() {
            Some(cap) if cap < 80 => {
                eprintln!(
                    "Warning: GPU compute capability {:.1} < 8.0 — bf16 Tensor Cores not available, falling back to f32.",
                    cap as f32 / 10.0
                );
                config.use_bf16 = false;
            }
            None => {
                eprintln!("Warning: could not detect GPU compute capability — disabling bf16 to be safe.");
                config.use_bf16 = false;
            }
            _ => {}
        }
    }
    let varmap = VarMap::new();
    let dtype = if config.use_bf16 { DType::BF16 } else { DType::F32 };
    println!("Model dtype: {:?}", dtype);
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let model = Model::new(dict.clone(), bpe, varmap, vb, device, config)?;

    Ok(model)
}

/// Load only the vocabulary and BPE rules from a saved model path,
/// without touching the weight file. Useful when constants have changed
/// and loading weights would cause a shape mismatch.
pub fn load_vocab(path: &str, device: &Device) -> Result<Model, std::io::Error> {
    let dict_path = format!("{}.dict", path);
    let file = fs::File::open(&dict_path)?;
    let dict_words: Vec<String> = serde_json::from_reader(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let dict = tokens_to_dict(dict_words);
    let bpe_path = format!("{}.bpe", path);
    let bpe = Bpe::load(&bpe_path).unwrap_or_else(|_| Bpe::new_empty());
    let config = {
        let config_path = format!("{}.config.json", path);
        fs::File::open(&config_path)
            .ok()
            .and_then(|f| serde_json::from_reader(f).ok())
            .unwrap_or_else(TrainConfig::from_env)
    };
    create_model(&dict, bpe, device, config).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
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

pub fn get_pretrained_dict(
    file_path: &str,
) -> Result<(Dict, Vec<String>, Bpe), candle_core::Error> {
    println!("Reading file: {}", file_path);
    let content = fs::read_to_string(file_path)?;
    println!("Read {} chars", content.len());

    let bpe = match Bpe::load("data/model.bpe") {
        Ok(bpe) => {
            println!(
                "Loaded BPE from data/model.bpe ({} merges, {} words cached)",
                bpe.merges.len(),
                bpe.vocab_size()
            );
            bpe
        }
        Err(_) => {
            let bpe = Bpe::learn(&content, NUM_BPE_MERGES);
            bpe.save("data/model.bpe")
                .unwrap_or_else(|e| eprintln!("Warning: could not save BPE: {}", e));
            bpe
        }
    };

    let tokens: Vec<String> = bpe.tokenize(&content);
    println!(
        "Dict size (before extras): {}",
        tokens_to_dict(tokens.clone()).len()
    );

    let lorem_tokens = bpe.tokenize("lorem ipsum et dolor sit amet");
    let hello_world_tokens = bpe.tokenize("hello world");
    let sys_tokens = vec![String::from(NOT_FOUND), String::from(STOP_TOKEN)];

    let tokens = [tokens, lorem_tokens, hello_world_tokens, sys_tokens].concat();

    let dict = tokens_to_dict(tokens.clone());
    println!("Dict size: {}", dict.len());

    return Ok((dict, tokens, bpe));
}
