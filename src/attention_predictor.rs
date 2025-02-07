use candle_core::{Device, Tensor, DType, D};
use candle_nn::{self as nn, Module};
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::token_utils::{Dict, GetTokenEmbedding, tokenize, EMBEDDING_SIZE};

pub struct Mlp {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub qs: Vec<nn::Linear>,
    pub ks: Vec<nn::Linear>,
    pub vs: Vec<nn::Linear>,
    pub var_map: VarMap,
    pub dict: Dict,
}

const CONTEXT_WINDOW: usize = 10;
const HIDDEN_SIZE: usize = 200;
const NUM_ATTENTION_HEADS: usize = 8;

impl Mlp {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder, ) -> Result<Self, candle_core::Error> {
        let mut qs: Vec<nn::Linear> = Vec::new();
        let mut ks: Vec<nn::Linear> = Vec::new();
        let mut vs: Vec<nn::Linear> = Vec::new();

        for i in 0..NUM_ATTENTION_HEADS {
            qs.push(nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, false, vb.pp(&format!("q{}", i)))?);
            ks.push(nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, false, vb.pp(&format!("k{}", i)))?);
            vs.push(nn::linear_b(HIDDEN_SIZE, HIDDEN_SIZE, false, vb.pp(&format!("v{}", i)))?);
        }

        let fc1 = nn::linear_b(HIDDEN_SIZE * NUM_ATTENTION_HEADS, EMBEDDING_SIZE * CONTEXT_WINDOW, false, vb.pp("fc1"))?;

        let fc2 = nn::linear_b(EMBEDDING_SIZE * CONTEXT_WINDOW, dict.len(), false, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2, qs, ks, vs, dict, var_map })
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, candle_core::Error> {
        let scale = 1.0 / ((HIDDEN_SIZE) as f64).sqrt();
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
        // position encoding
        let input = (input + self.position_encoding(input)?)?;

        let mut results: Vec<Tensor> = Vec::new();

        for i in 0..NUM_ATTENTION_HEADS {
            let q = input.apply(&self.qs[i])?;
            let k = input.apply(&self.ks[i])?;
            let v = input.apply(&self.vs[i])?;

            let result = self.scaled_dot_product_attention(&q, &k, &v)?;
            let input = nn::ops::dropout(&result, 0.4)?;
            let result = (((result * 0.7)? + input.clone())? * 0.3)?;

            results.push(result);
        }

        let result = Tensor::cat(&results, D::Minus1)?;

        // Add
        let result = self.fc1.forward(&result)?;
        let result = nn::ops::dropout(&result, 0.1)?;
        let result = result.relu()?;
        let result = self.fc2.forward(&result)?;

        //let result = nn::ops::softmax(&result, D::Minus1)?;
        let result = result.tanh()?;
        //let result = nn::ops::softmax(&result, D::Minus1)?;

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
}

pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, train: bool, device: &Device) -> Result<Mlp, candle_core::Error> {
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
    for (index, token) in tokens_chain.iter().enumerate() {
        if index < CONTEXT_WINDOW {
            continue;
        }

        let input_tokens: Vec<String> = tokens_chain[index - CONTEXT_WINDOW..index].to_vec();
        let input: Vec<f32> = input_tokens.iter().flat_map(|token| dict.get_token_embedding(token)).collect();
        let output: &String = token;

        let output_token_index: u32 = dict.get_word_index(output)?;

        let input = Tensor::new(input, &device)?;
        let target = one_hot(Tensor::new(output_token_index, &device)?, dict.len(), 1.0 as f32, 0.0 as f32)?;

        inputs.push(input);
        targets.push(target);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Mlp::new(dict, varmap, vb)?;

    // WARMUP
    // Optimizer settings
    // 1. More epoch when sample size is smaller
    let epochs = 8000;
    let initial_lr = 0.0001;
    let lr = initial_lr;
    let max_lr = initial_lr * 2.0;

    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(model.var_map.all_vars(), params)?;

    if !train {
        return Ok(model);
    }


    // Training loop
    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(&inputs)?;
        let lr = initial_lr + (max_lr - initial_lr) * (epoch as f64 / epochs as f64);
        optimizer.set_learning_rate(lr);

        // Compute loss
        // let loss = (&targets - &predictions)?.sqr()?.mean_all()?;
        // let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
        let loss = nn::loss::mse(&predictions, &targets)?;

        // Backpropagation
        optimizer.backward_step(&loss)?;

        if epoch % 100 == 0 {
            println!("Epoch {:6}: Loss = {:.6} Lr = {:.6}", epoch, loss.to_vec0::<f32>()?, lr);
            // print the first 10 predictions after "the horse"
            let input = "The horse ".to_string();
            let mut result = input.clone();
            for _ in 0..10 {
                let prediction = model.predict_next_token(result.as_str(), device)?;
                result = result + prediction.as_str();
            }
            println!("prediction: {}", result);
        }
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

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::token_utils::{tokenize, tokens_to_dict};

    use super::*;

    #[test]
    fn test_candle_predictor_hello_world() -> Result<(), candle_core::Error> {
        let tokens = tokenize("hello world");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("hello", &device)?, " ");
        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    fn test_candle_predictor_lorem() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("ipsum ", &device)?, "et");

        Ok(())
    }

        #[test]
    fn test_candle_predictor_lorem_2() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et dolor sit amet");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

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
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..10].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_20() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..20].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[ignore]
    #[test]
    fn test_horse_40() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..40].to_vec();

        let dict = tokens_to_dict(tokens.clone());

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
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..60].to_vec();

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
        // Define the file path
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
