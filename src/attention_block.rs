use candle_core::Tensor;
use candle_nn::{self as nn, Module};
use nn::VarBuilder;

use crate::attention_predictor::TRAINING_SUBSETS;

pub struct AttentionBlock {
    pub linear: Vec<nn::Linear>,
    pub qs: Vec<nn::Linear>,
    pub ks: Vec<nn::Linear>,
    pub vs: Vec<nn::Linear>,
    pub out_linear: nn::Linear,
    pub config: AttentionBlockConfig,
}

pub struct AttentionBlockConfig {
    pub input_size: usize,
    pub num_attention_heads: usize,
    pub context_window: usize,
    pub embedding_size: usize,
}

impl AttentionBlock {
    pub fn new(config: AttentionBlockConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let mut linear: Vec<nn::Linear> = Vec::new();
        let mut qs: Vec<nn::Linear> = Vec::new();
        let mut ks: Vec<nn::Linear> = Vec::new();
        let mut vs: Vec<nn::Linear> = Vec::new();

        let d_head = config.embedding_size / config.num_attention_heads;

        for i in 0..config.num_attention_heads {
            linear.push(nn::linear_b(
                d_head,
                d_head,
                true,
                vb.pp(&format!("linear{}", i)),
            )?);
            qs.push(nn::linear_b(
                d_head,
                d_head,
                true,
                vb.pp(&format!("q{}", i)),
            )?);
            ks.push(nn::linear_b(
                d_head,
                d_head,
                true,
                vb.pp(&format!("k{}", i)),
            )?);
            vs.push(nn::linear_b(
                d_head,
                d_head,
                true,
                vb.pp(&format!("v{}", i)),
            )?);
        }

        let out_linear = nn::linear_b(
            config.embedding_size,
            config.embedding_size,
            false,
            vb.pp("out_linear"),
        )?;

        Ok(Self {
            linear,
            qs,
            ks,
            vs,
            out_linear,
            config,
        })
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        // q, k, v: [batch, seq_len, d_head]
        let d_head = self.config.embedding_size / self.config.num_attention_heads;
        let scale = 1.0 / (d_head as f64).sqrt();

        // K^T: [batch, seq_len, d_head] -> [batch, d_head, seq_len]
        let k_t = k.transpose(1, 2)?.contiguous()?;

        // Q @ K^T: [batch, seq_len, seq_len]
        let scores = q.matmul(&k_t)?;
        let scores = (scores * scale)?;

        // Causal mask: lower triangle = 0.0, upper triangle = -inf
        let seq_len = self.config.context_window;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), q.device())?;

        // [batch, seq_len, seq_len] + [seq_len, seq_len] (broadcast over batch)
        let scores = scores.broadcast_add(&mask)?;

        let attn_weights = nn::ops::softmax(&scores, candle_core::D::Minus1)?;

        // [batch, seq_len, seq_len] @ [batch, seq_len, d_head] = [batch, seq_len, d_head]
        let result = attn_weights.matmul(&v)?;

        Ok(result)
    }

    pub fn position_encoding(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut position: Vec<f32> = Vec::new();

        for i in 0..self.config.context_window {
            for j in 0..self.config.embedding_size {
                let val = (i as f32)
                    / (10000_f32).powf(2.0 * (j as f32) / self.config.embedding_size as f32);
                if j % 2 == 0 {
                    position.push(val.sin());
                } else {
                    position.push(val.cos());
                }
            }
        }

        // [1, context_window, embedding_size] for broadcasting over batch
        let position = Tensor::from_slice(
            &position,
            (1, self.config.context_window, self.config.embedding_size),
            input.device(),
        )?;

        Ok(position)
    }

    pub fn forward(
        &self,
        input: &Tensor,
        train_subset_index: i8,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let batch_size = input.dim(0)?;
        let input_flat = input.clone();

        // Reshape [batch, context_window * embedding_size] -> [batch, context_window, embedding_size]
        let input = input.reshape((
            batch_size,
            self.config.context_window,
            self.config.embedding_size,
        ))?;

        let input = if train {
            nn::ops::dropout(&input, 0.1)?
        } else {
            input
        };

        let d_head = self.config.embedding_size / self.config.num_attention_heads;
        let mut results: Vec<Tensor> = Vec::new();

        for i in 0..self.config.num_attention_heads {
            // Extract this head's slice: [batch, seq_len, d_head]
            let start = i * d_head;
            let portions = input.narrow(2, start, d_head)?.contiguous()?;

            // Per-token linear + Q/K/V projections: [batch, seq_len, d_head]
            let linear_output = portions.apply(&self.linear[i])?;

            let q = linear_output.apply(&self.qs[i])?;
            let k = linear_output.apply(&self.ks[i])?;
            let v = linear_output.apply(&self.vs[i])?;

            // Causal attention: [batch, seq_len, d_head]
            let result = self.scaled_dot_product_attention(&q, &k, &v)?;

            let subset_size: i8 = self.config.num_attention_heads as i8 / TRAINING_SUBSETS;
            let current_subset_index = i as i8 / subset_size;

            if current_subset_index == train_subset_index {
                results.push(result);
            } else {
                let result = result.detach();
                results.push(result);
            }
        }

        // Concat heads: [batch, seq_len, embedding_size]
        let result = Tensor::cat(&results, 2)?;

        // Output projection per token: [batch, seq_len, embedding_size]
        let result = self.out_linear.forward(&result)?;

        // Flatten back: [batch, context_window * embedding_size]
        let result = result.reshape((batch_size, self.config.input_size))?;

        // Residual connection
        let result = (result + input_flat)?;

        Ok(result)
    }
}
