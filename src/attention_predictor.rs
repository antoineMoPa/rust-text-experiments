use std::{fs, io::Error, collections::BTreeMap};

use candle_core::{Device, Tensor, DType, D, Var};
use candle_nn::{self as nn, Module};
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::{token_utils::{tokenize, tokens_to_dict, Dict, GetTokenEmbedding, self}, read_n_chars, encoder_decoder::{EncoderDecoder, EMBEDDING_SIZE}, attention_block::{AttentionBlockConfig, AttentionBlock}};

// smoll
const CONTEXT_WINDOW: usize = 3;
const INPUT_SIZE: usize = EMBEDDING_SIZE * CONTEXT_WINDOW;
const NUM_ATTENTION_HEADS: usize = 4;
const ATTENTION_HEAD_INPUT_SIZE: usize =
    (EMBEDDING_SIZE / NUM_ATTENTION_HEADS)
    * CONTEXT_WINDOW;
const HIDDEN_SIZE: usize = 1024;
const NUM_BLOCKS: usize = 1;
pub const CHARS_TO_TRAIN_ON: usize = u64::pow(2, 17) as usize;
pub const FILE_PATH: &str = "data/corpus/level_0/corpus.corpus";

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

// Custom param-less LayerNorm implementation due to:
// Error: Metal error no metal implementation for layer-norm
fn layer_norm_no_params(x: &Tensor, eps: f64) -> Result<Tensor, candle_core::Error> {
    let mean = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let var = x.var(D::Minus1)?.unsqueeze(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;

    return (x.broadcast_sub(&mean))?.broadcast_div(&std);
}

pub struct Model {
    pub blocks: Vec<AttentionBlock>,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub var_map: VarMap,
    pub dict: Dict,
    pub encdec: EncoderDecoder,
    pub token_index: BTreeMap<String, u32>,
    pub token_embedding_map: BTreeMap<String, Vec<f32>>,
    pub token_embedding_tensor_map: BTreeMap<String, Tensor>,
}

impl Model {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder, device: &Device) -> Result<Self, candle_core::Error> {
        let mut blocks = Vec::new();

        for b in 0..NUM_BLOCKS {
            let config: AttentionBlockConfig = AttentionBlockConfig {
                attention_head_input_size: ATTENTION_HEAD_INPUT_SIZE,
                input_size: INPUT_SIZE,
                num_attention_heads: NUM_ATTENTION_HEADS,
                context_window: CONTEXT_WINDOW,
                embedding_size: EMBEDDING_SIZE,
            };

            let block = AttentionBlock::new(config, vb.push_prefix(&format!("block_{}", b)))?;
            blocks.push(block);
        }

        let fc1 = nn::linear_b(INPUT_SIZE, HIDDEN_SIZE, true, vb.pp("fc1"))?;
        let fc2 = nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, true, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(HIDDEN_SIZE, EMBEDDING_SIZE, true, vb.pp("fc3"))?;

        let encdec = EncoderDecoder::load_from_path("data/encdec", &device)?;

        let token_index = dict.build_index();
        let token_embedding_map: BTreeMap<String, Vec<f32>> = encdec.build_token_embedding_map()?;
        let token_embedding_tensor_map: BTreeMap<String, Tensor> = encdec.build_token_embedding_tensor_map()?;

        Ok(Self {
            fc1,
            fc2,
            fc3,
            blocks,
            dict,
            var_map,
            encdec,
            token_index,
            token_embedding_map,
            token_embedding_tensor_map,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut result = self.blocks[0].forward(&input)?;

        for (index, block) in self.blocks.iter().enumerate() {
            if index == 0 {
                continue;
            }
            result = block.forward(&result)?;
        }

        let result = (result + (input * 0.2)?)?;
        let result = (result * 0.2)?;

        let result = self.fc1.forward(&result)?;
        let result = self.fc2.forward(&result)?;
        let result = self.fc3.forward(&result)?;

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

        let output_prob = self.encdec.unembed(&output)?;

        // let output_prob = (output_prob / 10.0)?;

        let output_prob_max_index = output_prob.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn run_str(&self, input: &str, len: usize, device: &Device) -> Result<String, candle_core::Error> {
        let mut output = String::new();
        let tokens = tokenize(input);
        let mut input: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            let result = self.encdec.get_token_embedding_vec(token.as_str())?;
            input.push(result);
        }

        for _ in 0..len {
            let prediction = self.run(&input, device)?;
            let token_embedding = self.encdec.get_token_embedding_vec(prediction.as_str())?;
            input.push(token_embedding);
            output.push_str(prediction.as_str());
        }

        return Ok(output);
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

    pub fn is_good(&self, test_str: &str, device: &Device) -> bool {
        let mut is_good = true;
        let tokens = tokenize(test_str);
        let mut prediction_str = String::new();
        for i in 1..(tokens.len()) {
            let prediction = self.predict_next_token(tokens[0..i].join("").as_str(), device).unwrap();

            prediction_str = prediction_str + prediction.as_str();

            if prediction != tokens[i] {
                is_good = false;
                break;
            }
        }

        println!("Predicted: {}", prediction_str);

        return is_good;
    }

    pub fn simple_training_loop(&mut self, inputs: Tensor, targets: Tensor, lr: f64, epochs: u32) -> Result<(), candle_core::Error> {
        let mut optimizer = candle_nn::AdamW::new_lr(self.var_map.all_vars(), lr)?;
        let mut epoch: u32 = 0;

        while epoch < epochs {
            // Forward pass
            let predictions = self.forward(&inputs)?;

            // Compute loss
            // binary cross-entropy is not supported on metal gpu
            // let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
            let loss = nn::loss::mse(&predictions, &targets)?;

            // Backpropagation
            optimizer.backward_step(&loss)?;

            if epoch > 0 && epoch % 20 == 0 {
                println!("Epoch {:6}: Loss = {:.6} Lr = {:.6}", epoch, loss.to_vec0::<f32>()?, lr);

            }
            epoch += 1;
        }

        Ok(())
    }


    pub fn good_bad_training_loop(&mut self, inputs: Tensor, targets: Tensor, test_str: &str, epochs: u32, device: &Device) -> Result<(), candle_core::Error> {
        // 1. More epoch when sample size is smaller
        let initial_lr = 0.00002;
        let lr = initial_lr;
        let max_lr = initial_lr * 3.0;

        let params = ParamsAdamW {
            lr,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(self.var_map.all_vars(), params)?;

        let mut good_index: i32 = -1;

        // Training loop
        let mut epoch: u32 = 0;
        let mut total_good_yet: u32 = 0;
        let mut amount_good: u32 = 0;
        let mut amount_bad: u32 = 0;

        while epoch < epochs {
            // Forward pass
            let predictions = self.forward(&inputs)?;
            let lr = initial_lr + (max_lr - initial_lr) * (epoch as f64 / epochs as f64);
            optimizer.set_learning_rate(lr);

            // Compute loss
            // binary cross-entropy is not supported on metal gpu
            // let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
            let loss = nn::loss::mse(&predictions, &targets)?;

            // Backpropagation
            optimizer.backward_step(&loss)?;

            if epoch % 10 == 0 {
                let is_good = self.is_good(test_str, device);
                let is_good_str = if is_good { "GOOD" } else { "BAD" };
                println!("Epoch {:6}: Loss = {:.6} Lr = {:.6} [{}]", epoch, loss.to_vec0::<f32>()?, lr, is_good_str);

                if is_good {
                    self.save_to_path("data/good");
                    total_good_yet += 1;
                    good_index = epoch as i32;
                    amount_bad = 0;
                    amount_good += 1;

                    if amount_good >= 5 {
                        println!("Model, is good 5 times - stopping training.");
                        break;
                    }
                }
                else {
                    amount_bad += 1;
                    amount_good = 0;

                    let should_load_last_good =
                        epoch > 50 &&
                        good_index != -1 &&
                        (amount_bad > total_good_yet &&
                         amount_bad >= 5);

                    if should_load_last_good {
                        println!("loading good model, going back to epoch {}", good_index);
                        self.load_inplace_from_path("data/good")?;
                        // epoch = good_index as u32;
                        amount_bad = 0;
                    }
                }
            }
            epoch += 1;
        }

        if !self.is_good(test_str, device) && good_index != -1 {
            println!("model is BAD");
            println!("loading last good model, going back to epoch {}", good_index);
            self.var_map.load("data/good.safetensors")?;
        }


        Ok(())
    }

    pub fn gen_training_data(&mut self, tokens_chain: Vec<String>, device: &Device) -> Result<(Tensor, Tensor), candle_core::Error> {
        let mut inputs: Vec<Tensor> = Vec::new();
        let mut targets: Vec<Tensor> = Vec::new();

        // pad token chain with context window zeros
        let mut padding: Vec<String> = Vec::new();
        let mut tokens_chain = tokens_chain.clone();

        for _ in 0..CONTEXT_WINDOW {
            padding.push(" ".to_string());
            tokens_chain.insert(0, " ".to_string());
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

            inputs.push(input);
            targets.push(target.squeeze(0)?);
        }

        let inputs = Tensor::stack(&inputs, 0)?;
        let targets = Tensor::stack(&targets, 0)?;

        return Ok((inputs, targets));
    }

    pub fn train(&mut self, tokens_chain: Vec<String>, epochs: u32, test_str: &str, device: &Device) -> Result<(), candle_core::Error> {

        let (inputs, targets) = self.gen_training_data(tokens_chain, device)?;

        self.good_bad_training_loop(inputs, targets, test_str, epochs, device)?;

        Ok(())
    }

    pub fn simple_train(&mut self, tokens_chain: Vec<String>, device: &Device) -> Result<(), candle_core::Error> {
        let token_batch_size = 80;
        let epochs: u32 = 10;
        let num_batches = tokens_chain.len() / token_batch_size;
        let lr = 1e-5;
        let mut optimizer = candle_nn::AdamW::new_lr(self.var_map.all_vars(), lr)?;

        for epoch in 0..epochs {
            for j in 0..(num_batches + 1) {
                let start = j * token_batch_size;
                let overlap = 10;
                let end = start + token_batch_size + overlap;
                let end = end.min(tokens_chain.len());
                let tokens = tokens_chain[start..end].to_vec();

                let (inputs, targets) = self.gen_training_data(tokens, device)?;

                let predictions = self.forward(&inputs)?;

                // Compute loss
                // binary cross-entropy is not supported on metal gpu
                //let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
                let loss = nn::loss::mse(&predictions, &targets)?;

                //let sigmoid_predictions =
                //    (((predictions/2.0)?.tanh()? + 1.0)? / 2.0)?;

                //let loss = nn::loss::mse(&sigmoid_predictions, &targets)?;

                // Backpropagation
                optimizer.backward_step(&loss)?;

                let loss_stat = loss.to_vec0::<f32>()?;
                //}

                if j % 2 == 0 && j > 0 {
                    println!("Epoch {:6}: Loss = {:.6}", epoch, loss_stat);
                    println!("Batch        {:6} / {}", j, num_batches);
                    let prediction = self.run_str(" ", 10, device)?;
                    println!("Prediction: {}", prediction);
                }
            }
            self.save_to_path("data/model");
        }

        self.print_stats()?;

        Ok(())
    }

    pub fn print_stats(&self) -> Result<(), candle_core::Error> {
        println!("Model stats:");
        println!("Dict size: {}", self.dict.len());

        // print min, max, mean, std of all tensors
        for var in self.var_map.all_vars().iter() {
            let min = var.flatten_all()?.min(D::Minus1)?.to_vec0::<f32>()?;
            let max = var.flatten_all()?.max(D::Minus1)?.to_vec0::<f32>()?;
            let mean = var.flatten_all()?.mean(D::Minus1)?.to_vec0::<f32>()?;
            let variance = var.flatten_all()?.var(D::Minus1)?.to_vec0::<f32>()?;
            println!("{}: min: {:.3}, max: {:.3}, mean: {:.3}, std: {:.3}", "", min, max, mean, variance);
        }

        Ok(())
    }

    pub fn save_to_path(&self, path: &str) {
        let var_map_path = format!("{}.safetensors", path);
        self.var_map.save(var_map_path.as_str()).unwrap();

        let dict_words = self.dict.iter().map(|(word, _)| word.clone()).collect::<Vec<String>>();

        let dict_path = format!("{}.dict", path);
        let file = fs::File::create(dict_path).unwrap();
        serde_json::to_writer(file, &dict_words).unwrap();
    }

    pub fn load_from_path(path: &str, device: &Device) -> Result<Self, Error> {

        let dict_path = format!("{}.dict", path);
        let file = fs::File::open(dict_path).unwrap();
        let dict_words: Vec<String> = serde_json::from_reader(file).unwrap();

        let dict = tokens_to_dict(dict_words);

        let mut model = create_model(dict, device).unwrap();

        let var_map_path = format!("{}.safetensors", path);
        model.var_map.load(var_map_path.as_str()).unwrap();

        Ok(model)
    }

    pub fn load_inplace_from_path(&mut self, path: &str) -> Result<(), Error> {
        let dict_path = format!("{}.dict", path);
        let file = fs::File::open(dict_path).unwrap();
        let dict_words: Vec<String> = serde_json::from_reader(file).unwrap();

        let dict = tokens_to_dict(dict_words);
        self.dict = dict;

        let var_map_path = format!("{}.safetensors", path);
        self.var_map.load(var_map_path.as_str()).unwrap();

        Ok(())
    }
}

pub fn create_model(dict: Dict, device: &Device) -> Result<Model, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Model::new(dict.clone(), varmap, vb, device)?;

    Ok(model)
}


pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, train: bool, device: &Device) -> Result<Model, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = Model::new(dict.clone(), varmap, vb, device)?;

    if train {
        model.train( tokens_chain, 400, &"The horse", device)?;
    }

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_predictor_hello_world() -> Result<(), candle_core::Error> {
        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        let test_str = "hello world";
        let tokens = tokenize(test_str);
        model.train(tokens, 250, test_str,  &device)?;

        assert_eq!(model.predict_next_token("hello", &device)?, " ");
        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    fn test_candle_predictor_lorem() -> Result<(), candle_core::Error> {
        let test_str = "lorem ipsum et";

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        let tokens = tokenize(test_str);
        model.train(tokens, 500, test_str,  &device)?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("lorem ipsum ", &device)?, "et");

        Ok(())
    }

    #[test]
    fn test_candle_predictor_lorem_2() -> Result<(), candle_core::Error> {
        let test_str = "lorem ipsum et dolor sit amet";

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        let tokens = tokenize(test_str);
        model.train(tokens, 500, test_str,  &device)?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("ipsum ", &device)?, "et");
        assert_eq!(model.predict_next_token("dolor ", &device)?, "sit");

        assert_eq!(model.predict_next_token("lorem ipsum", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ipsum ", &device)?, "et");
        assert_eq!(model.predict_next_token("lorem ipsum et ", &device)?, "dolor");
        assert_eq!(model.predict_next_token("ipsum et ", &device)?, "dolor");

        Ok(())
    }

    #[test]
    fn test_horse_10() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let tokens = tokens[0..10].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        model.train(tokens, 500, test_str.as_str(),  &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_20() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let tokens = tokens[0..20].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        model.train( tokens, 500, test_str.as_str(),  &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_40() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let tokens = tokens[0..40].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        model.train( tokens.clone(), 500, test_str.as_str(),  &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[35..39].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[39]);

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_60() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let tokens = tokens[0..60].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        model.train( tokens.clone(), 500, test_str.as_str(),  &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[51..56].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[56]);

        let substring = tokens[51..57].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[57]);

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_100() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let tokens = tokens[0..100].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;
        let mut model = Model::load_from_path("data/model", &device)?;

        model.train( tokens.clone(), 200, test_str.as_str(),  &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);

        let substring = tokens[63..70].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[70]);

        Ok(())
    }
}
