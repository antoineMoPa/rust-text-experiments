use std::{fs, io::Error};

use candle_core::{Device, Tensor, DType, D};
use candle_nn::{self as nn, Module};
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::token_utils::{tokenize, tokens_to_dict, Dict, GetTokenEmbedding, EMBEDDING_SIZE};

pub struct Mlp {
    pub linear: Vec<nn::Linear>,
    pub qs: Vec<nn::Linear>,
    pub ks: Vec<nn::Linear>,
    pub vs: Vec<nn::Linear>,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub var_map: VarMap,
    pub dict: Dict,
}

const CONTEXT_WINDOW: usize = 10;
const INPUT_SIZE: usize = EMBEDDING_SIZE * CONTEXT_WINDOW;
const NUM_ATTENTION_HEADS: usize = 8;
const HIDDEN_SIZE: usize = 4096;

impl Mlp {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder, ) -> Result<Self, candle_core::Error> {
        let mut linear: Vec<nn::Linear> = Vec::new();
        let mut qs: Vec<nn::Linear> = Vec::new();
        let mut ks: Vec<nn::Linear> = Vec::new();
        let mut vs: Vec<nn::Linear> = Vec::new();

        for i in 0..NUM_ATTENTION_HEADS {
            linear.push(nn::linear_b(INPUT_SIZE, INPUT_SIZE, false, vb.pp(&format!("linear{}", i)))?);
            qs.push(nn::linear_b(INPUT_SIZE, INPUT_SIZE, false, vb.pp(&format!("q{}", i)))?);
            ks.push(nn::linear_b(INPUT_SIZE, INPUT_SIZE, false, vb.pp(&format!("k{}", i)))?);
            vs.push(nn::linear_b(INPUT_SIZE, INPUT_SIZE, false, vb.pp(&format!("v{}", i)))?);
        }

        let fc1 = nn::linear_b(INPUT_SIZE * NUM_ATTENTION_HEADS, HIDDEN_SIZE, false, vb.pp("fc1"))?;

        let fc2 = nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, false, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(HIDDEN_SIZE, dict.len(), false, vb.pp("fc3"))?;

        Ok(Self { linear, qs, ks, fc1, fc2, fc3, vs, dict, var_map })
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, candle_core::Error> {
        let scale = 1.0 / ((INPUT_SIZE) as f64).sqrt();
        let result = q.matmul(&k.t()?)?;
        let result = (result * scale)?;
        let result = nn::ops::softmax(&result, D::Minus1)?;
        let result = result.matmul(&v)?;

        Ok(result)
    }

    fn position_encoding(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut position: Vec<f32> = Vec::new();

        for i in 0..CONTEXT_WINDOW {
            for j in 0..EMBEDDING_SIZE {
                let val = (i as f32) / (10000 as f32).powf(2.0 * (j as f32) / EMBEDDING_SIZE as f32);
                if i % 2 == 0 {
                    position.push(val.sin());
                } else {
                    position.push(val.cos());
                }
            }
        }

        let position = Tensor::new(position, input.device())?;

        let batch_size = input.dim(0)?;
        let encoding = position.repeat(&[batch_size, 1])?;

        return Ok(encoding);
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let input = (input + self.position_encoding(input)?)?;
        let input = nn::ops::dropout(&input, 0.15)?;

        let mut results: Vec<Tensor> = Vec::new();

        for i in 0..NUM_ATTENTION_HEADS {
            let linear_output = input.apply(&self.linear[i])?;

            let linear_output = nn::ops::dropout(&linear_output, 0.1)?;

            let q = linear_output.apply(&self.qs[i])?;
            let k = linear_output.apply(&self.ks[i])?;
            let v = linear_output.apply(&self.vs[i])?;

            let result = self.scaled_dot_product_attention(&q, &k, &v)?;

            // would love to use norm instead here, but it's not supported on metal gpu
            let result = (((result * 0.7)? + input.clone())? * 0.3)?;

            results.push(result);
        }

        let result = Tensor::cat(&results, D::Minus1)?;

        let result = self.fc1.forward(&result)?;
        let result = result.relu()?;
        let result = self.fc2.forward(&result)?;
        let result = result.relu()?;
        let result = self.fc3.forward(&result)?;

        let result = result.tanh()?;

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

        let output_prob = self.forward(&input)?;
        let output_prob = (output_prob / 10.0)?;

        let output_prob_max_index = output_prob.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn predict_next_token(&self, input: &str, device: &Device) -> Result<String, candle_core::Error> {
        let tokens = tokenize(&input);
        let mut input: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            input.push(self.dict.get_token_embedding(token.as_str()));
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

    pub fn simple_training_loop(&mut self, inputs: Tensor, targets: Tensor, initial_lr: f64, epochs: u32) -> Result<(), candle_core::Error> {
        // 1. More epoch when sample size is smaller
        let lr = initial_lr;
        let max_lr = initial_lr * 6.0;

        let params = ParamsAdamW {
            lr,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(self.var_map.all_vars(), params)?;
        let mut epoch: u32 = 0;

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

            if epoch % 20 == 0 {
                println!("Epoch {:6}: Loss = {:.6} Lr = {:.6}", epoch, loss.to_vec0::<f32>()?, lr);

            }
            epoch += 1;
        }

        Ok(())
    }


    pub fn good_bad_training_loop(&mut self, inputs: Tensor, targets: Tensor, test_str: &str, epochs: u32, device: &Device) -> Result<(), candle_core::Error> {
        // 1. More epoch when sample size is smaller
        let initial_lr = 0.00008;
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
            let token = tokens_chain[index].clone();
            let  mut input_tokens: Vec<String>;

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

            let input: Vec<f32> = input_tokens.iter().flat_map(|token| self.dict.get_token_embedding(token)).collect();
            let output: String = token.clone();

            let output_token_index: u32 = self.dict.get_word_index(&output)?;

            let input = Tensor::new(input, &device)?;
            let target = one_hot(Tensor::new(output_token_index, &device)?, self.dict.len(), 0.95 as f32, 0.0 as f32)?;

            inputs.push(input);
            targets.push(target);
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


    pub fn simple_train(&mut self, tokens_chain: Vec<String>, epochs: u32, initial_lr: f64, device: &Device) -> Result<(), candle_core::Error> {
        let token_batch_size = 60;
        let num_batches = tokens_chain.len() / token_batch_size;

        for i in 0..epochs {
            println!("Global epoch {}/{}", i, epochs);
            for j in 0..num_batches {
                let start = j * token_batch_size;
                let end = (j + 1) * token_batch_size;
                let tokens = tokens_chain[start..end].to_vec();

                let (inputs, targets) = self.gen_training_data(tokens, device)?;

                self.simple_training_loop(inputs, targets, initial_lr, 4)?;
            }
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

        let mut mlp = create_model(dict, device).unwrap();

        let var_map_path = format!("{}.safetensors", path);
        mlp.var_map.load(var_map_path.as_str()).unwrap();

        Ok(mlp)
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

pub fn create_model(dict: Dict, device: &Device) -> Result<Mlp, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Mlp::new(dict.clone(), varmap, vb)?;

    Ok(model)
}


pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, train: bool, device: &Device) -> Result<Mlp, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = Mlp::new(dict.clone(), varmap, vb)?;

    if train {
        model.train( tokens_chain, 400, &"The horse", device)?;
    }

    Ok(model)
}


pub fn get_device() -> Result<Device, candle_core::Error> {
    let device = Device::new_metal(0)?;
    match &device {
        Device::Metal(m) => m,
        _ => panic!("Device is not Metal"),
    };
    return Ok(device);
}

pub fn get_pretrained_dict() -> Result<(Dict, Vec<String>), candle_core::Error> {
    let file_path = "data/corpus/wiki-horse.txt";
    let content = fs::read_to_string(file_path)?;
    let tokens: Vec<String> = tokenize(&content)[0..1000].to_vec();
    let lorem_tokens = tokenize("lorem ipsum et dolor sit amet");
    let hello_world_tokens = tokenize("hello world");

    let tokens = [tokens, lorem_tokens, hello_world_tokens].concat();

    let dict = tokens_to_dict(tokens.clone());

    return Ok((dict, tokens));
}


#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_candle_predictor_hello_world() -> Result<(), candle_core::Error> {
        let (dict, _tokens) = get_pretrained_dict()?;

        let device = get_device()?;

        let mut model = create_model(dict, &device)?;
        model.var_map.load("data/horse_pretrain.safetensors")?;

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

        let (dict, _tokens) = get_pretrained_dict()?;

        let device = get_device()?;

        let mut model = create_model(dict, &device)?;
        model.var_map.load("data/horse_pretrain.safetensors")?;

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

        let (dict, _tokens) = get_pretrained_dict()?;

        let device = get_device()?;

        let mut model = create_model(dict, &device)?;
        model.var_map.load("data/horse_pretrain.safetensors")?;

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
        let (dict, tokens) = get_pretrained_dict()?;
        let tokens = tokens[0..10].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;

        let mut model = create_model(dict, &device)?;
        model.var_map.load("data/horse_pretrain.safetensors")?;

        model.train(tokens, 500, test_str.as_str(),  &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_20() -> Result<(), candle_core::Error> {
        let (dict, tokens) = get_pretrained_dict()?;
        let tokens = tokens[0..20].to_vec();
        let test_str = tokens[0..4].to_vec().join("");

        let device = get_device()?;

        let mut model = create_model(dict, &device)?;
        model.var_map.load("data/horse_pretrain.safetensors")?;

        model.train( tokens, 500, test_str.as_str(),  &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_40() -> Result<(), candle_core::Error> {
        let (dict, tokens) = get_pretrained_dict()?;
        let tokens = tokens[0..40].to_vec();

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[35..39].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[39]);

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_60() -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict()?;
        let tokens = tokens[0..40].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

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
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..100].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);

        let substring = tokens[63..70].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[70]);

        Ok(())
    }

    // #[test]
    // fn test_horse_200() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..200].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //     let substring = tokens[102..113].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[113]);
    //
    //     let substring = tokens[102..114].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);
    //
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_horse_400() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..400].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //
    //     let substring = tokens[102..114].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);
    //
    //     let substring = tokens[162..182].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[182]);
    //
    //     let substring = tokens[190..211].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[211]);
    //
    //     let substring = tokens[330..341].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);
    //
    //     let substring = tokens[330..342].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[342]);
    //
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_horse_1000() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..1000].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //     let substring = tokens[330..341].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);
    //
    //     let substring = tokens[810..831].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[831]);
    //
    //     let substring = tokens[810..832].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[832]);
    //
    //     Ok(())
    // }
}
