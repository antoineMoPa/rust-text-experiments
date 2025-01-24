use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot};

use crate::token_utils::{Dict, GetTokenEmbedding, EMBEDDING_SIZE};

pub struct Mlp {
    fc1: nn::Linear,
    act: candle_nn::Activation,
    fc2: nn::Linear,
    dict: Dict,
}

impl Mlp {
    pub fn new(vb: VarBuilder, dict: Dict) -> Result<Self, candle_core::Error> {
        let fc1 = nn::linear(EMBEDDING_SIZE, 32,vb.pp("fc1"))?;
        let fc2 = nn::linear(32, dict.len(),vb.pp("fc2"))?;

        let act = candle_nn::activation::Activation::Relu;
        Ok(Self { fc1, fc2, act, dict })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.act)?
            .apply(&self.fc2)?
            .apply(&nn::activation::Activation::Relu)
    }

    pub fn run(&self, input: &Vec<f32>, device: &Device) -> Result<String, candle_core::Error> {
        let input = Tensor::new(input.clone(), device)?.unsqueeze(0)?;

        let output_prob = self.forward(&input)?;
        let output_prob_max_index = output_prob.argmax(1)?;
        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }
}

fn create_and_train_autoencoder_model(dict: Dict) -> Result<Mlp, candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    let mut inputs: Vec<Tensor> = Vec::new();
    let mut targets: Vec<Tensor> = Vec::new();

    for (index, (word, _)) in dict.iter().enumerate() {
        let input = Tensor::new(dict.get_token_embedding(word), &device)?;
        let token_index: u32 = index as u32;
        let target = one_hot(Tensor::new(token_index, &device)?, dict.len(), 1.0 as f32, 0.0 as f32)?;

        inputs.push(input);
        targets.push(target);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    // Create the XORNet model
    let model = Mlp::new(vb, dict)?;

    // Optimizer settings
    let params = ParamsAdamW {
        lr: 0.01,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop
    for epoch in 0..200 {
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

    use crate::{token_utils::{tokenize, tokens_to_dict}, embedding_utils::get_token_embedding};

    use super::*;

    #[test]
    fn test_candle_autoencoder_hello_world() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("hello, world!");
        let dict = tokens_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = create_and_train_autoencoder_model(dict).unwrap();

        let hello_embedding = get_token_embedding("hello", &model.dict);
        assert!(model.run(&hello_embedding, &device)? == "hello");

        let world_embedding = get_token_embedding("world", &model.dict);
        assert!(model.run(&world_embedding, &device)? == "world");

        // Different words should have different results
        assert!(model.run(&hello_embedding, &device)? != model.run(&world_embedding, &device)?);

        Ok(())
    }

    #[test]
    fn test_encode_larger_vocabulary() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("This is a longer string, hello, world!");
        let dict = tokens_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = create_and_train_autoencoder_model(dict).unwrap();

        let this_embedding = get_token_embedding("This", &model.dict);
        assert!(model.run(&this_embedding, &device)? == "This");

        let is_embedding = get_token_embedding("is", &model.dict);
        assert!(model.run(&is_embedding, &device)? == "is");

        let a_embedding = get_token_embedding("a", &model.dict);
        assert!(model.run(&a_embedding, &device)? == "a");

        let longer_embedding = get_token_embedding("longer", &model.dict);
        assert!(model.run(&longer_embedding, &device)? == "longer");

        let string_embedding = get_token_embedding("string", &model.dict);
        assert!(model.run(&string_embedding, &device)? == "string");

        let comma_embedding = get_token_embedding(",", &model.dict);
        assert!(model.run(&comma_embedding, &device)? == ",");

        let hello_embedding = get_token_embedding("hello", &model.dict);
        assert!(model.run(&hello_embedding, &device)? == "hello");

        let world_embedding = get_token_embedding("world", &model.dict);
        assert!(model.run(&world_embedding, &device)? == "world");

        let exclamation_embedding = get_token_embedding("!", &model.dict);
        assert!(model.run(&exclamation_embedding, &device)? == "!");

        // Different words should have different embeddings
        assert!(model.run(&this_embedding, &device)? != model.run(&world_embedding, &device)?);

        Ok(())
    }
}
