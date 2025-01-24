use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::token_utils::{Dict, GetTokenEmbedding, tokenize};

pub struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dict: Dict,
    embedding_size: u32,
}

const CONTEXT_WINDOW: usize = 3;

impl Mlp {
    pub fn new(vb: VarBuilder, embedding_size: u32, dict: Dict) -> Result<Self, candle_core::Error> {
        let hidden_size = 128;
        let fc1 = nn::linear(embedding_size as usize * CONTEXT_WINDOW, hidden_size,vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden_size, dict.len(),vb.pp("fc2"))?;

        Ok(Self { fc1, fc2, dict, embedding_size })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.fc2)?
            .apply(&nn::activation::Activation::Sigmoid)
    }

    pub fn run(&self, input_embedding: &Vec<Vec<f64>>, device: &Device) -> Result<String, candle_core::Error> {
        let embedding_size = self.embedding_size;

        let mut input: Vec<f64> = Vec::new();

        if input_embedding.len() > CONTEXT_WINDOW {
            // ... then add the input
            let start = input_embedding.len() - CONTEXT_WINDOW;
            let end = start + CONTEXT_WINDOW;
            input.append(&mut input_embedding[start..end].to_vec().concat());
        }
        else {
            // pad with zeros
            input = vec![0.0; (CONTEXT_WINDOW - input_embedding.len()) * embedding_size as usize];
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
        let mut input: Vec<Vec<f64>> = Vec::new();

        for token in tokens {
            input.push(self.dict.get_token_embedding(token.as_str()));
        }

        self.run(&input, device)
    }
}

fn create_and_train_predictor_model(dict: Dict, embedding_size: u32, tokens_chain: Vec<String>) -> Result<Mlp, candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

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
        let input: Vec<f64> = input_tokens.iter().flat_map(|token| dict.get_token_embedding(token)).collect();

        let output: &String = token;

        let output_token_index: u32 = dict.get_word_index(output)?;

        let input = Tensor::new(input, &device)?;
        let target = one_hot(Tensor::new(output_token_index, &device)?, dict.len(), 1.0, 0.0)?;

        inputs.push(input);
        targets.push(target);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);

    // Create the XORNet model
    let model = Mlp::new(vb, embedding_size, dict)?;

    // Optimizer settings
    let mut epoch = 200;
    let mut lr = 0.02;

    if tokens_chain.len() < 30 {
        epoch = 200;
        lr = 0.01;
    }

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
        let loss = (&predictions - &targets)?.sqr()?.mean_all()?;
        //let loss = nn::loss::binary_cross_entropy_with_logit(&predictions, &targets)?;

        // Backpropagation
        optimizer.backward_step(&loss)?;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        }
    }

    Ok(model)
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

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens)?;

        assert_eq!(model.predict_next_token("hello", &device)?, " ");
        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    fn test_candle_predictor_lorem() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et");

        let dict = tokens_to_dict(tokens.clone());

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens.clone())?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("ipsum ", &device)?, "et");

        Ok(())
    }

        #[test]
    fn test_candle_predictor_lorem_2() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et dolor sit amet");

        let dict = tokens_to_dict(tokens.clone());

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens.clone())?;

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

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens)?;

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

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens)?;

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

        let device = Device::Cpu;

        let model = create_and_train_predictor_model(dict, 2, tokens.clone())?;

        let substring = tokens[35..38].to_vec().join(" ");

        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        Ok(())
    }
}
