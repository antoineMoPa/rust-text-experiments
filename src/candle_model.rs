use std::io::Error;

use rand::Rng;

use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW};

struct Mlp {
    fc1: nn::Linear,
    act: candle_nn::Activation,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let fc1 = nn::linear(8, 4,vb.pp("fc1"))?;
        let act = candle_nn::activation::Activation::Relu;

        Ok(Self { fc1, act })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.act)
    }
}

fn build_and_train_model() -> Result<(), candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..4000 {
        // Generate random sample
        let mut sample = Vec::new();
        for _ in 0..8 {
            sample.push(rand::thread_rng().gen_range(0..2) as f64);
        }
        let mut sample_result = Vec::new();
        // XOR pairs of values
        for i in 0..4 {
            let a = sample[(i * 2)] as i32;
            let b = sample[(i * 2) + 1] as i32;
            let result = a ^ b;

            sample_result.push(result as f64);
        }

        println!("Sample: {:?}, Result: {:?}", sample, sample_result);

        inputs.push(Tensor::new(&[
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
        ], &device)?);
        targets.push(Tensor::new(&[
            sample_result[0],
            sample_result[1],
            sample_result[2],
            sample_result[3],
        ], &device)?);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);

    // Create the XORNet model
    let model = Mlp::new(vb)?;

    // Optimizer settings
    let params = ParamsAdamW {
        lr: 0.02,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let predictions = model.forward(&inputs)?;

        // Compute loss (mean squared error)
        let loss = (&predictions - &targets)?.sqr()?.mean_all()?;

        // Backpropagation
        optimizer.backward_step(&loss)?;

        //if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        //}
    }

    // Test the model
    let inputs = Tensor::new(&[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]], &device)?;
    let targets = Tensor::new(&[[0.0, 1.0, 1.0, 0.0]], &device)?;
    let test_preds = model.forward(&inputs)?;
    println!("Predictions: {:?}", test_preds.to_string());

    Ok(())
}




fn build_candle_model() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_basic() {
        build_candle_model().unwrap();
    }

    #[test]
    fn test_candle_xor() {
        build_and_train_model().unwrap();
    }
}
