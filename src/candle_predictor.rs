use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::token_utils::{Dict, GetTokenEmbedding, tokenize, EMBEDDING_SIZE};

pub struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dict: Dict,
}

const CONTEXT_WINDOW: usize = 10;

impl Mlp {
    pub fn new(vb: VarBuilder, dict: Dict) -> Result<Self, candle_core::Error> {
        let hidden_size = 256;

        let fc1 = nn::linear(EMBEDDING_SIZE as usize * CONTEXT_WINDOW, hidden_size,vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden_size, dict.len(),vb.pp("fc2"))?;

        Ok(Self { fc1, fc2, dict })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let result = input;
        let result = result.apply(&self.fc1)?;
        // let result = self.bn.forward_t(&result, false)?;
        let result = nn::ops::dropout(&result, 0.4)?;
        let result = result.relu()?;
        let result = result.apply(&self.fc2)?;
        // let result = result.tanh()?;
        let result = nn::ops::softmax(&result, 1);

        return result;
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

pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, device: &Device) -> Result<Mlp, candle_core::Error> {
    // Define training data for XOR
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

    // Create the XORNet model
    let model = Mlp::new(vb, dict)?;

    // Optimizer settings
    // 1. More epoch when sample size is smaller
    let epoch = 100 + 3000 / tokens_chain.len();
    let lr = 0.008;

    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop

    for epoch in 0..epoch {
        // Forward pass
        let predictions = model.forward(&inputs)?;

        // Compute loss
        // let loss = (&targets - &predictions)?.sqr()?.mean_all()?;
        // let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;
        let loss = nn::loss::mse(&predictions, &targets)?;

        // Backpropagation
        optimizer.backward_step(&loss)?;

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        }
    }

    Ok(model)
}

pub fn get_device() -> Result<Device, candle_core::Error> {
    let device = Device::new_metal(0)?;
    let metal_device = match &device {
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

        let model = create_and_train_predictor_model(dict, tokens, &device)?;

        assert_eq!(model.predict_next_token("hello", &device)?, " ");
        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    fn test_candle_predictor_lorem() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

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

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

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

        let model = create_and_train_predictor_model(dict, tokens, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_20() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..20].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_40() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..40].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[35..39].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[39]);

        Ok(())
    }

    #[test]
    fn test_horse_60() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..60].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[51..56].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[56]);

        let substring = tokens[51..57].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[57]);

        Ok(())
    }

    #[test]
    fn test_horse_100() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..100].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);

        let substring = tokens[63..70].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[70]);

        Ok(())
    }

    #[test]
    fn test_horse_200() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..200].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);


        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);

        let substring = tokens[102..113].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[113]);

        let substring = tokens[102..114].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);

        Ok(())
    }

    #[test]
    fn test_horse_400() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..400].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);


        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);


        let substring = tokens[102..114].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);

        let substring = tokens[162..182].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[182]);

        let substring = tokens[190..211].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[211]);

        let substring = tokens[330..341].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);

        let substring = tokens[330..342].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[342]);

        Ok(())
    }

        #[test]
    fn test_horse_1000() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..1000].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let model = create_and_train_predictor_model(dict, tokens.clone(), &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);


        let substring = tokens[63..69].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);

        let substring = tokens[330..341].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);

        let substring = tokens[810..831].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[831]);

        let substring = tokens[810..832].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[832]);

        Ok(())
    }

}
