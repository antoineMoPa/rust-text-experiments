use std::io::Error;

use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW};

struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(device: Device) -> Self {
        let weight = Tensor::randn(0.5, 0.5, (8, 8), &device.clone()).unwrap();
        let fc1 = nn::Linear::new(weight, None);

        let weight = Tensor::randn(0.5, 0.5, (4, 8), &device.clone()).unwrap();
        let fc2 = nn::Linear::new(weight, None);

        Self { fc1, fc2 }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input
            .apply(&self.fc1)?
            .apply(&self.fc2)
    }
}

// Define a simple feedforward network
struct XORNet {
    mlp: Mlp,
}

impl XORNet {
    fn new(device: &Device) -> Result<Self, Error> {
        let mlp = Mlp::new(device.clone());

        Ok(Self { mlp })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.mlp.forward(input)
    }
}

fn build_and_train_model() -> Result<(), candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    // let inputs = Tensor::new(&[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]], &device)?;
    // let targets = Tensor::new(&[[0.0, 1.0, 1.0, 0.0]], &device)?;

    // Define more training data for XOR
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..100 {
        inputs.push(Tensor::new(&[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]], &device)?);
        targets.push(Tensor::new(&[[0.0, 1.0, 1.0, 0.0]], &device)?);
    }

    let inputs = Tensor::stack(&inputs, 0)?;
    let targets = Tensor::stack(&targets, 0)?;

    // Create the XORNet model
    let mut model = XORNet::new(&device)?;

    // Optimizer settings
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let params = ParamsAdamW {
        lr: 0.0001,
        ..Default::default()
    };
    let varmap = VarMap::new();
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // Training loop
    for epoch in 0..1000 {
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
