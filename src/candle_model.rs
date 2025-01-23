use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW};

use crate::embedding_utils::get_token_embedding;

pub struct Mlp {
    fc1: nn::Linear,
    act: candle_nn::Activation,
    fc2: nn::Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder, embedding_size: u32) -> Result<Self, candle_core::Error> {
        let fc1 = nn::linear(embedding_size as usize, 32,vb.pp("fc1"))?;
        let fc2 = nn::linear(32, embedding_size as usize,vb.pp("fc2"))?;

        let act = candle_nn::activation::Activation::Relu;
        Ok(Self { fc1, fc2, act })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.act)?
            .apply(&self.fc2)?
            .apply(&nn::activation::Activation::Sigmoid)
    }

    pub fn run(&self, input: &Vec<f64>, device: &Device) -> Result<Vec<f64>, candle_core::Error> {
        println!("Running model with input: {:?}", input);
        let input = Tensor::new(input.clone(), device)?.unsqueeze(0)?;

        println!("output: {:?}", self.forward(&input)?.to_vec2::<f64>()?[0]);

        Ok(self.forward(&input)?.to_vec2::<f64>()?[0].clone())
    }
}

fn build_model(embedding_size: u32, examples: &Vec<Vec<f64>>) -> Result<Mlp, candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    let mut inputs: Vec<Tensor> = Vec::new();
    let mut targets: Vec<Tensor> = Vec::new();

    for example in examples {
        let input = Tensor::new(example.clone(), &device)?;
        let target = input.clone();

        inputs.push(input);
        targets.push(target);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);

    // Create the XORNet model
    let model = Mlp::new(vb, embedding_size)?;

    // Optimizer settings
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop
    for epoch in 0..200 {
        // Forward pass
        let predictions = model.forward(&inputs)?;

        // Compute loss (mean squared error)
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

pub fn create_and_train_model_for_dict(dict: &std::collections::HashMap<String, f64>, embed_size: u32) -> Result<Mlp, candle_core::Error> {
    let mut examples: Vec<Vec<f64>> = Vec::new();

    for (_i, token) in dict.iter().enumerate() {
        let token_embedding = get_token_embedding(token.0, dict);
        examples.push(token_embedding);
    }

    build_model(embed_size, &examples)
}


#[cfg(test)]
mod tests {
    use crate::{token_utils::{tokenize, vocabulary_to_dict}, embedding_utils::are_embeddings_close};

    use super::*;

    #[test]
    fn test_candle_encoder_decoder_hello_world() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = create_and_train_model_for_dict(&dict, 2).unwrap();

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(are_embeddings_close(&model.run(&hello_embedding, &device)?, &hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(&model.run(&world_embedding, &device)?, &world_embedding, 0.15));

        // Different words should have different results
        assert!(!are_embeddings_close(&model.run(&hello_embedding, &device)?, &world_embedding, 0.15));

        Ok(())
    }
}
