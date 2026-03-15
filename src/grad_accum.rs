use candle_core::{DType, Result, Tensor, Var};

pub struct AdamW {
    vars: Vec<Var>,
    first_moment: Vec<Tensor>,
    second_moment: Vec<Tensor>,
    step_t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let vars: Vec<Var> = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        // Always keep optimizer states in F32 for numerical stability (mixed-precision).
        let first_moment = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), DType::F32, v.device()))
            .collect::<Result<Vec<_>>>()?;
        let second_moment = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), DType::F32, v.device()))
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
        })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;

        self.step_t += 1;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let scale_m = 1.0 / (1.0 - beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.step_t as i32));

        // Gradient clipping: compute global norm across all parameters.
        const CLIP_NORM: f64 = 1.0;
        let mut global_norm_sq = 0f64;
        for (i, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                let g = grad.to_dtype(DType::F32)?;
                let norm_sq = g.sqr()?.sum_all()?.to_vec0::<f32>()? as f64;
                global_norm_sq += norm_sq;
                // Stash the f32 grad temporarily — recompute below (simpler than caching).
                let _ = (i, norm_sq);
            }
        }
        let clip_scale = (CLIP_NORM / global_norm_sq.sqrt()).min(1.0);

        for (i, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                // Cast gradient to F32 — moments are always F32 (mixed-precision training).
                let g = (grad.to_dtype(DType::F32)? * clip_scale)?;

                let m = &self.first_moment[i];
                let v = &self.second_moment[i];

                let next_m = ((m * beta1)? + (&g * (1.0 - beta1))?)?;
                let next_v = ((v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;

                // Compute update in F32, then cast back to the var's dtype (e.g. BF16).
                let theta_f32 = var.as_tensor().to_dtype(DType::F32)?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
                let next_theta = ((theta_f32 * (1.0 - lr * self.weight_decay))? - (adjusted_grad * lr)?)?;

                self.first_moment[i] = next_m;
                self.second_moment[i] = next_v;
                var.set(&next_theta.to_dtype(var.dtype())?)?;
            }
        }

        Ok(())
    }
}
