use std::{fs, io::Error, collections::BTreeMap};

use candle_core::{Device, Tensor, DType, D};
use candle_nn::{self as nn, Module};
use nn::{VarMap, Optimizer, VarBuilder};
use colored::Colorize;
use crate::models::RunStr;
use crate::token_utils::STOP_TOKEN;
use crate::{token_utils::{tokenize, tokens_to_dict, Dict}, read_n_chars, encoder_decoder::{EncoderDecoder, EMBEDDING_SIZE}, attention_block::{AttentionBlockConfig, AttentionBlock}};

// smoll
const DOWNSCALE_FACTOR: usize = 3;
const CONTEXT_WINDOW: usize = 32;
const INPUT_SIZE: usize = EMBEDDING_SIZE * CONTEXT_WINDOW;
const NUM_ATTENTION_HEADS: usize = 21;
const HIDDEN_SIZE: usize = 2048;
const NUM_BLOCKS: usize = 2;
pub const CHARS_TO_TRAIN_ON: usize = u64::pow(2, 17) as usize;
pub const FILE_PATH: &str = "data/corpus/level_3/corpus.corpus";
const LR: f64 = 3.0e-4;
const EPOCHS: u32 = 30;
const TOKEN_BATCH_SIZE: usize = 128;

// large
// const INPUT_SIZE: usize = EMBEDDING_SIZE * CONTEXT_WINDOW;
// const NUM_ATTENTION_HEADS: usize = 8;
// const ATTENTION_HEAD_INPUT_SIZE: usize = (EMBEDDING_SIZE / NUM_ATTENTION_HEADS) * CONTEXT_WINDOW;
// const CONTEXT_WINDOW: usize = 40;
// const HIDDEN_SIZE: usize = 4096;
// const NUM_BLOCKS: usize = 10;
// pub const CHARS_TO_TRAIN_ON: usize = u64::pow(2, 15) as usize;
// const FILE_PATH: &str = "data/corpus/blogtext.csv";


const NOT_FOUND: &str = "<notfound>";

pub struct Model {
    pub blocks: Vec<AttentionBlock>,
    pub downscaler: nn::Linear,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub fc4: nn::Linear,
    pub var_map: VarMap,
    pub encdec: EncoderDecoder,
    pub token_embedding_map: BTreeMap<String, Vec<f32>>,
    pub token_embedding_tensor_map: BTreeMap<String, Tensor>,
    pub device: Device,
    pub train_subset_index: i8,
}

impl Model {
    pub fn new(var_map: VarMap, vb: VarBuilder, device: &Device) -> Result<Self, candle_core::Error> {
        if EMBEDDING_SIZE % NUM_ATTENTION_HEADS != 0 {
            // List out possible num attention heads
            for i in 1..(EMBEDDING_SIZE / 2) {
                if EMBEDDING_SIZE % i == 0 {
                    println!("Possible num attention heads for this embedding size: {}", i);
                }
            }
            // List out possible embeddding size
            for i in 1..10 {
                println!("Possible embedding size for this num attention heads: {}", i * NUM_ATTENTION_HEADS);
            }

            panic!("The embedding size should be divisible by the number of attention heads!");
        }

        let mut blocks = Vec::new();

        for b in 0..NUM_BLOCKS {
            let config: AttentionBlockConfig = AttentionBlockConfig {
                input_size: INPUT_SIZE / DOWNSCALE_FACTOR,
                num_attention_heads: NUM_ATTENTION_HEADS,
                context_window: CONTEXT_WINDOW,
                embedding_size: EMBEDDING_SIZE / DOWNSCALE_FACTOR,
                output_size: INPUT_SIZE / DOWNSCALE_FACTOR,
            };

            let block = AttentionBlock::new(config, vb.push_prefix(&format!("block_{}", b)))?;
            blocks.push(block);
        }

        let downscaler = nn::linear_b(INPUT_SIZE, INPUT_SIZE / DOWNSCALE_FACTOR, true, vb.pp("downscaler"))?;
        let fc1 = nn::linear_b(INPUT_SIZE / DOWNSCALE_FACTOR, HIDDEN_SIZE, true, vb.pp("fc1"))?;
        let fc2 = nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, true, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(HIDDEN_SIZE, EMBEDDING_SIZE, true, vb.pp("fc3"))?;
        let fc4 = nn::linear_b(EMBEDDING_SIZE, EMBEDDING_SIZE, true, vb.pp("fc4"))?;

        let encdec = EncoderDecoder::load_from_path("data/encdec", &device)?;

        let token_embedding_map: BTreeMap<String, Vec<f32>> = encdec.build_token_embedding_map()?;
        let token_embedding_tensor_map: BTreeMap<String, Tensor> = encdec.build_token_embedding_tensor_map()?;

        // Print the following info in
        println!("Embedding Size, Context Window, Epochs, Hidden Size, Num blocks, Num att. heads, LR");
        let output = format!("{}, {}, {}, {}, {}, {}, {}, {}, {}", encdec.dict.len(), EMBEDDING_SIZE, CONTEXT_WINDOW, EPOCHS, HIDDEN_SIZE, NUM_BLOCKS, NUM_ATTENTION_HEADS, LR, TOKEN_BATCH_SIZE);
        println!("{}", output.on_white().black());


        Ok(Self {
	    downscaler,
            fc1,
            fc2,
            fc3,
            fc4,
            blocks,
            var_map,
            encdec,
            token_embedding_map,
            token_embedding_tensor_map,
            device: device.clone(),
            train_subset_index: 0,
        })
    }

    fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.train_subset_index = (self.train_subset_index + 1) % 2;

        return self.forward(input);
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut result = self.downscaler.forward(&input)?;
        let downscaled_input = result.clone();

        for block in self.blocks.iter() {
            result = block.forward(&result)?.tanh()?;
        }

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after attention blocks");
        }

        let result = (result + (downscaled_input * 0.2)?)?;
        let result = (result * 0.2)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after input addition");
        }

        let result = self.fc1.forward(&result)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after fc1");

            let w = self.fc1.weight();
            let b = self.fc1.bias().unwrap();

            let i: f32 = input.abs()?.max_all()?.to_vec0()?;
            println!("i = {:?}", i);
            let r: f32 = result.abs()?.max_all()?.to_vec0()?;
            println!("r = {:?}", r);
            let w: f32 = w.abs()?.max_all()?.to_vec0()?;
            println!("w = {:?}", w);
            let b: f32 = b.abs()?.max_all()?.to_vec0()?;
            println!("b = {:?}", b);
        }

        let result = self.fc2.forward(&result)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after fc2");
        }

        let result = self.fc3.forward(&result)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after fc3");
        }

        // Re-use encdec layer
        //let result = result.gelu()?;
        let result = self.encdec.fc2.forward(&result)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after encdec fc2");
        }

        let result = result.tanh()?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after tanh");
        }

        let result = self.fc4.forward(&result)?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after fc4");
        }

        let result = result.tanh()?;

        if result.sum_all()?.to_vec0::<f32>()?.is_nan() {
            println!("Result is nan after final tanh");
        }

        return Ok(result);
    }

    pub fn run(&self, input_embedding: &Vec<Vec<f32>>, device: &Device) -> Result<String, candle_core::Error> {
        let mut input: Vec<f32> = Vec::new();

        if input_embedding.len() > CONTEXT_WINDOW {
            // ... then add the input
            let start = input_embedding.len() - CONTEXT_WINDOW;
            let end = start + CONTEXT_WINDOW;
            input.append(&mut input_embedding[start..end].to_vec().concat());
        }
        else {
            // pad with zeros
            input = vec![0.0; (CONTEXT_WINDOW - input_embedding.len()) * EMBEDDING_SIZE as usize];
            input.append(&mut input_embedding.to_vec().concat());
        }

        let input = Tensor::new(input, device)?.unsqueeze(0)?;

        let output = self.forward(&input)?;

        let output = self.encdec.unembed(&output)?;

        let output = (output / 2.0)?;

        let output_prob_max_index = output.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.encdec.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn predict_next_token(&self, input: &str, device: &Device) -> Result<String, candle_core::Error> {
        let tokens = tokenize(&input);
        let mut input: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            match self.token_embedding_map.get(token.as_str()) {
                Some(embedding) => input.push(embedding.clone()),
                None => {
                    input.push(self.token_embedding_map.get(NOT_FOUND).unwrap().clone());
                },
            }
        }

        self.run(&input, device)
    }

    pub fn gen_training_data(&mut self, tokens_chain: Vec<String>, device: &Device) -> Result<(Tensor, Tensor), candle_core::Error> {
        let mut inputs: Vec<Tensor> = Vec::new();
        let mut targets: Vec<Tensor> = Vec::new();

        // pad token chain with context window zeros
        let tokens_chain = tokens_chain.clone();

        for _ in 0..CONTEXT_WINDOW {
            //padding.push(" ".to_string());
            //tokens_chain.insert(0, " ".to_string());
        }

        // iterate over tokens_chain
        for index in 0..(tokens_chain.len()) {
            let target_token = tokens_chain[index].clone();
            let mut input_tokens: Vec<String>;

            if index == 0 {
                continue;
            }

            if index < CONTEXT_WINDOW {
                input_tokens = tokens_chain[0..index].to_vec();
                let mut padding: Vec<String> = Vec::new();

                for _i in 0..CONTEXT_WINDOW-index {
                    padding.push(String::from(" "));
                }

                input_tokens = [padding, input_tokens].concat();
            } else {
                input_tokens = tokens_chain[index - CONTEXT_WINDOW..index].to_vec();
            }

            // debug input and output
            // println!("input: {:?}, output: {:?}", input_tokens, target_token);

            let input: Vec<f32> = input_tokens.iter().flat_map(|token| self.token_embedding_map.get(token).unwrap().to_vec()).collect();


            let input = Tensor::new(input, &device)?;
            let target = self.token_embedding_tensor_map.get(target_token.as_str());
            let target = match target {
                Some(t) => t.clone(),
                None => {
                    println!("Token not found: {}", target_token);
                    self.token_embedding_tensor_map.get(NOT_FOUND).unwrap().clone()
                },
            };

            // Add some noise to input
            let input_shape = input.shape().clone();
            let input = (input + Tensor::randn( 0.0 as f32, 0.00625 as f32, input_shape, device)?)?;

            inputs.push(input);
            targets.push(target.squeeze(0)?);
        }

        let inputs = Tensor::stack(&inputs, 0)?;
        let targets = Tensor::stack(&targets, 0)?;

        return Ok((inputs, targets));
    }

    pub fn tensor_to_token(&self, tensor: &Tensor) -> Result<String, candle_core::Error> {
        let output = self.encdec.unembed(&tensor)?;

        let output = (output / 2.0)?;

        let output_prob_max_index = output.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.encdec.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn tensors_to_tokens(&self, tokens: &Vec<Tensor>) -> Result<Vec<String>, candle_core::Error> {
        Ok(tokens.into_iter()
            .map(
                | t:& Tensor |
                self.tensor_to_token(&t.unsqueeze(0).unwrap()).unwrap()
            ).collect())
    }

    pub fn crash_dump(&self, inputs: Tensor, targets: Tensor) -> Result<(), candle_core::Error> {
        self.save_to_path("data/model");

        let batch_size = inputs.dim(0)?;
        self.print_stats()?;

        println!("Batch size {}", batch_size);

        let inputs = inputs.chunk(batch_size, 0)?;
        let targets = targets.chunk(batch_size, 0)?;


        for i in 0..batch_size {
            let input = inputs[i].squeeze(0)?.chunk(CONTEXT_WINDOW, 0)?;
            println!(
                "batch[{}] {:?} -> {:?}",
                i,
                self.tensors_to_tokens(&input)?,
                self.tensor_to_token(&targets[i])?
            );
        }

        return Ok(());
    }

    pub fn simple_train(&mut self, tokens_chain: Vec<String>, device: &Device) -> Result<(), candle_core::Error> {
        let token_batch_size = TOKEN_BATCH_SIZE;
        let epochs: u32 = EPOCHS;
        let num_batches = tokens_chain.len() / token_batch_size + 1;
        let lr = LR;
        let mut optimizer = candle_nn::AdamW::new_lr(self.var_map.all_vars(), lr)?;

        for epoch in 0..epochs {
            let mut loss_stat: f32 = 1.0;

            for j in 0..(num_batches + 1) {
                let start = j * token_batch_size;
                let overlap = 0;
                let end = start + token_batch_size + 1 + overlap;
                let end = end.min(tokens_chain.len());
                if end <= start {
                    continue;
                }
                let tokens = tokens_chain[start..end].to_vec();

                let (inputs, targets) = self.gen_training_data(tokens, device)?;

                let predictions = self.forward_train(&inputs)?;

                // Compute loss
                 // binary cross-entropy is not supported on metal gpu
                //let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
                let loss = nn::loss::mse(&predictions, &targets)?;
                let batch_size = end - start;
                let factor = 20.0 / batch_size as f64;
                optimizer.set_learning_rate(LR * factor);

                loss_stat = loss.to_vec0::<f32>()?;

                if loss_stat.is_nan() {
                    self.crash_dump(inputs, targets)?;
                    panic!("Loss is nan, gradient probably exploded or vanished.");
                }

                optimizer.backward_step(&loss)?;
            }

            println!("Epoch {:6}/{:6} : Loss = {:.6}", epoch, epochs, loss_stat);
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

            if epoch % 40 == 0 {
                self.save_to_path("data/model");
            }
        }

        self.print_stats()?;

        Ok(())
    }

    pub fn print_stats(&self) -> Result<(), candle_core::Error> {
        println!("Model stats:");
        println!("Dict size: {}", self.encdec.dict.len());

        // print min, max, mean, std of all tensors
        for var in self.var_map.all_vars().iter() {
            let min = var.min_all()?.to_vec0::<f32>()?;
            let max = var.max_all()?.to_vec0::<f32>()?;
            let mean = var.mean_all()?.to_vec0::<f32>()?;
            let variance = var.flatten_all()?.var(D::Minus1)?.to_vec0::<f32>()?;
            println!("{}: min: {:.3}, max: {:.3}, mean: {:.3}, std: {:.3}", "", min, max, mean, variance);
        }

        Ok(())
    }

    pub fn save_to_path(&self, path: &str) {
        let var_map_path = format!("{}.safetensors", path);
        self.var_map.save(var_map_path.as_str()).unwrap();

        let dict_words = self.encdec.dict.iter().map(|(word, _)| word.clone()).collect::<Vec<String>>();

        let dict_path = format!("{}.dict", path);
        let file = fs::File::create(dict_path).unwrap();
        serde_json::to_writer(file, &dict_words).unwrap();
    }

    pub fn load_from_path(path: &str, device: &Device) -> Result<Self, Error> {

        let mut model = create_model(device).unwrap();

        let var_map_path = format!("{}.safetensors", path);
        model.var_map.load(var_map_path.as_str()).unwrap();

        Ok(model)
    }
}

impl RunStr for Model {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error> {
        let mut output = String::new();
        let tokens = tokenize(input);
        let mut input: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            let result = self.encdec.get_token_embedding_vec(token.as_str())?;
            input.push(result);
        }

        for _ in 0..len {
            let prediction = self.run(&input, &self.device)?;
            let token_embedding = self.encdec.get_token_embedding_vec(prediction.as_str())?;
            input.push(token_embedding);

            if prediction == STOP_TOKEN {
                break;
            }

            output.push_str(prediction.as_str());
        }

        return Ok(output);
    }
}

pub fn create_model(device: &Device) -> Result<Model, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Model::new(varmap, vb, device)?;

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

pub fn get_pretrained_dict(file_path: &str) -> Result<(Dict, Vec<String>), candle_core::Error> {
    // Experiments with char count
    // exponent 15+ makes no sense, no spaces, etc.
    // exponent 14+ has some spaces after 8 iterations
    let char_count = CHARS_TO_TRAIN_ON as u64;

    println!("Reading {} chars from file: {}", char_count, file_path);
    let content = read_n_chars(file_path, char_count)?;
    println!("Read {} chars", content.len());
    let tokens: Vec<String> = tokenize(&content).to_vec();
    let dict = tokens_to_dict(tokens.clone());
    println!("Dict size: {}", dict.len());

    let lorem_tokens = tokenize("lorem ipsum et dolor sit amet");
    let hello_world_tokens = tokenize("hello world");
    let sys_tokens = vec![String::from(NOT_FOUND)];

    let tokens = [tokens, lorem_tokens, hello_world_tokens, sys_tokens].concat();

    let dict = tokens_to_dict(tokens.clone());

    return Ok((dict, tokens));
}
