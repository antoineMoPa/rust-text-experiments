use candle_core::{Result, Tensor, Var};
use std::collections::HashMap;

/// AdamW optimizer with gradient accumulation support.
///
/// Allows calling `accumulate()` multiple times (one per micro-batch),
/// then `step()` once to apply the averaged gradients.
pub struct AccumAdamW {
    vars: Vec<Var>,
    first_moment: Vec<Tensor>,
    second_moment: Vec<Tensor>,
    step_t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    accumulated_grads: HashMap<usize, Tensor>,
    accum_count: usize,
}

impl AccumAdamW {
    pub fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let vars: Vec<Var> = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        let first_moment = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), v.dtype(), v.device()))
            .collect::<Result<Vec<_>>>()?;
        let second_moment = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), v.dtype(), v.device()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vars,
            first_moment,
            second_moment,
            step_t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            accumulated_grads: HashMap::new(),
            accum_count: 0,
        })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Compute gradients for a loss and add them to the accumulator.
    pub fn accumulate(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;

        for (i, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                let entry = self.accumulated_grads.remove(&i);
                let new_grad = match entry {
                    Some(existing) => (existing + grad)?,
                    None => grad.clone(),
                };
                self.accumulated_grads.insert(i, new_grad);
            }
        }
        self.accum_count += 1;

        Ok(())
    }

    /// Apply the averaged accumulated gradients with AdamW update rule, then clear.
    pub fn step(&mut self) -> Result<()> {
        if self.accum_count == 0 {
            return Ok(());
        }

        self.step_t += 1;
        let lr = self.lr;
        let lr_lambda = lr * self.weight_decay;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let scale_m = 1.0 / (1.0 - beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.step_t as i32));
        let accum_count = self.accum_count as f64;

        for (i, var) in self.vars.iter().enumerate() {
            if let Some(grad) = self.accumulated_grads.get(&i) {
                let g = (grad / accum_count)?;

                let m = &self.first_moment[i];
                let v = &self.second_moment[i];

                let next_m = ((m * beta1)? + (&g * (1.0 - beta1))?)?;
                let next_v = ((v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (var.as_tensor() * (1.0 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;

                self.first_moment[i] = next_m;
                self.second_moment[i] = next_v;
                var.set(&next_theta)?;
            }
        }

        self.accumulated_grads.clear();
        self.accum_count = 0;

        Ok(())
    }
}
