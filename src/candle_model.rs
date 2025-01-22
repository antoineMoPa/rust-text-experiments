use rand::Rng;

use candle_core::{Device, Tensor, DType};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW};

struct Mlp {
    fc1: nn::Linear,
    act: candle_nn::Activation,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let fc1 = nn::linear(8, 32,vb.pp("fc1"))?;
        let fc2 = nn::linear(32, 4,vb.pp("fc2"))?;

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
}

fn build_and_train_model() -> Result<Mlp, candle_core::Error> {
    // Use the default device (CPU in this case)
    let device = Device::Cpu;

    // Define training data for XOR
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..200 {
        // Generate random sample
        let mut sample = Vec::new();
        for _ in 0..8 {
            sample.push(rand::thread_rng().gen_range(0..2) as f64);
        }
        let mut sample_result = Vec::new();
        // XOR pairs of values
        for i in 0..4 {
            let a = sample[i * 2] as i32;
            let b = sample[(i * 2) + 1] as i32;
            let result = a ^ b;

            sample_result.push(result as f64);
        }

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

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        }
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_xor() {
        let device = Device::Cpu;
        let model = build_and_train_model().unwrap();

        let inputs = Tensor::new(&[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]], &device).unwrap();
        let targets = Tensor::new(&[[0.0, 1.0, 1.0, 0.0]], &device).unwrap();
        let test_preds = model.forward(&inputs).unwrap();

        let diff: f64 = (test_preds - &targets).unwrap().sum_all().unwrap().to_vec0().unwrap();
        assert!(diff < 0.03);
    }
}
