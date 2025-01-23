use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::{embedding_utils::get_token_embedding, token_utils::{Dict, GetTokenEmbedding}};

pub struct Mlp {
    fc1: nn::Linear,
    act: candle_nn::Activation,
    fc2: nn::Linear,
    dict: Dict,
}

impl Mlp {
    pub fn new(vb: VarBuilder, embedding_size: u32, dict: Dict) -> Result<Self, candle_core::Error> {
        let fc1 = nn::linear(embedding_size as usize, 32,vb.pp("fc1"))?;
        let fc2 = nn::linear(32, dict.len(),vb.pp("fc2"))?;

        let act = candle_nn::activation::Activation::Relu;
        Ok(Self { fc1, fc2, act, dict })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.act)?
            .apply(&self.fc2)?
            .apply(&nn::activation::Activation::Sigmoid)
    }

    pub fn run(&self, input: &Vec<f64>, device: &Device) -> Result<String, candle_core::Error> {
        let input = Tensor::new(input.clone(), device)?.unsqueeze(0)?;

        let output_prob = self.forward(&input)?;
        let output_prob_max_index = output_prob.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }
}

fn build_autoencoder_model(embedding_size: u32, dict: Dict) -> Result<Mlp, candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    let mut inputs: Vec<Tensor> = Vec::new();
    let mut targets: Vec<Tensor> = Vec::new();

    for (index, (word, _)) in dict.iter().enumerate() {
        let input = Tensor::new(dict.get_token_embedding(word), &device)?;
        let token_index: u32 = index as u32;
        let target = one_hot(Tensor::new(token_index, &device)?, dict.len(), 1.0, 0.0)?;

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
    let params = ParamsAdamW {
        lr: 0.03,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop
    for epoch in 0..300 {
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

pub fn create_and_train_autoencoder_model(dict: &Dict, embed_size: u32) -> Result<Mlp, candle_core::Error> {
    build_autoencoder_model(embed_size, dict.clone())
}


#[cfg(test)]
mod tests {
    use std::fs;

    use crate::token_utils::{tokenize, vocabulary_to_dict};

    use super::*;

    #[test]
    fn test_candle_autoencoder_hello_world() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = create_and_train_autoencoder_model(&dict, 2).unwrap();

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(model.run(&hello_embedding, &device)? == "hello");

        let world_embedding = get_token_embedding("world", &dict);
        assert!(model.run(&world_embedding, &device)? == "world");

        // Different words should have different results
        assert!(model.run(&hello_embedding, &device)? != model.run(&world_embedding, &device)?);

        Ok(())
    }

    #[test]
    fn test_encode_larger_vocabulary() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("This is a longer string, hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = create_and_train_autoencoder_model(&dict, 2).unwrap();

        let this_embedding = get_token_embedding("This", &dict);
        assert!(model.run(&this_embedding, &device)? == "This");

        let is_embedding = get_token_embedding("is", &dict);
        assert!(model.run(&is_embedding, &device)? == "is");

        let a_embedding = get_token_embedding("a", &dict);
        assert!(model.run(&a_embedding, &device)? == "a");

        let longer_embedding = get_token_embedding("longer", &dict);
        assert!(model.run(&longer_embedding, &device)? == "longer");

        let string_embedding = get_token_embedding("string", &dict);
        assert!(model.run(&string_embedding, &device)? == "string");

        let comma_embedding = get_token_embedding(",", &dict);
        assert!(model.run(&comma_embedding, &device)? == ",");

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(model.run(&hello_embedding, &device)? == "hello");

        let world_embedding = get_token_embedding("world", &dict);
        assert!(model.run(&world_embedding, &device)? == "world");

        let exclamation_embedding = get_token_embedding("!", &dict);
        assert!(model.run(&exclamation_embedding, &device)? == "!");

        // Different words should have different embeddings
        assert!(model.run(&this_embedding, &device)? != model.run(&world_embedding, &device)?);

        Ok(())
    }

    #[test]
    fn test_candle_autoencoder_horse() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[..100].to_vec();

        let dict = vocabulary_to_dict(tokens);

        let device = Device::Cpu;
        let model = create_and_train_autoencoder_model(&dict, 2)?;

        let horse_embedding = get_token_embedding("horse", &dict);
        assert!(model.run(&horse_embedding, &device)? == "horse");

        Ok(())
    }
}
