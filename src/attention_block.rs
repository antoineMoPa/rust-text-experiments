use candle_core::Tensor;
use candle_nn::{self as nn, Module};
use nn::VarBuilder;

pub struct AttentionBlock {
    pub qs: Vec<nn::Linear>,
    pub ks: Vec<nn::Linear>,
    pub vs: Vec<nn::Linear>,
    pub out_linear: nn::Linear,
    pub ffn_in: nn::Linear,
    pub ffn_out: nn::Linear,
    pub config: AttentionBlockConfig,
    causal_mask: Tensor,
    pos_enc: Tensor,
}

pub struct AttentionBlockConfig {
    pub input_size: usize,
    pub num_attention_heads: usize,
    pub context_window: usize,
    pub embedding_size: usize,
    pub ffn_hidden: usize,
}

impl AttentionBlock {
    pub fn new(config: AttentionBlockConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let mut qs: Vec<nn::Linear> = Vec::new();
        let mut ks: Vec<nn::Linear> = Vec::new();
        let mut vs: Vec<nn::Linear> = Vec::new();

        let d_head = config.embedding_size / config.num_attention_heads;

        for i in 0..config.num_attention_heads {
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

        let ffn_in = nn::linear_b(config.embedding_size, config.ffn_hidden, true, vb.pp("ffn_in"))?;
        let ffn_out = nn::linear_b(config.ffn_hidden, config.embedding_size, true, vb.pp("ffn_out"))?;

        let device = vb.device();
        let seq_len = config.context_window;

        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let causal_mask = Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?;

        let mut pe_data: Vec<f32> = Vec::with_capacity(seq_len * config.embedding_size);
        for i in 0..seq_len {
            for j in 0..config.embedding_size {
                let val = (i as f32)
                    / (10000_f32).powf(2.0 * (j as f32) / config.embedding_size as f32);
                pe_data.push(if j % 2 == 0 { val.sin() } else { val.cos() });
            }
        }
        let pos_enc = Tensor::from_slice(&pe_data, (1, seq_len, config.embedding_size), device)?;

        Ok(Self {
            qs,
            ks,
            vs,
            out_linear,
            ffn_in,
            ffn_out,
            config,
            causal_mask,
            pos_enc,
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

        // [batch, seq_len, seq_len] + [seq_len, seq_len] (broadcast over batch)
        let scores = scores.broadcast_add(&self.causal_mask)?;

        let attn_weights = nn::ops::softmax(&scores, candle_core::D::Minus1)?;

        // [batch, seq_len, seq_len] @ [batch, seq_len, d_head] = [batch, seq_len, d_head]
        let result = attn_weights.matmul(&v)?;

        Ok(result)
    }

    pub fn position_encoding(&self) -> &Tensor {
        &self.pos_enc
    }

    pub fn forward(
        &self,
        input: &Tensor,
        train: bool,
    ) -> Result<Tensor, candle_core::Error> {
        let batch_size = input.dim(0)?;

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

            // Q/K/V projections: [batch, seq_len, d_head]
            let q = portions.apply(&self.qs[i])?;
            let k = portions.apply(&self.ks[i])?;
            let v = portions.apply(&self.vs[i])?;

            // Causal attention: [batch, seq_len, d_head]
            let result = self.scaled_dot_product_attention(&q, &k, &v)?;
            results.push(result);
        }

        // Concat heads: [batch, seq_len, embedding_size]
        let result = Tensor::cat(&results, 2)?;

        // Output projection per token: [batch, seq_len, embedding_size]
        let result = self.out_linear.forward(&result)?;

        // Residual connection (still in [batch, seq, emb])
        let result = (result + input.reshape((batch_size, self.config.context_window, self.config.embedding_size))?)?;

        // Per-token FFN sublayer with residual
        let ffn_residual = result.clone();
        let result = self.ffn_in.forward(&result)?.gelu()?;
        let result = self.ffn_out.forward(&result)?;
        let result = (result + ffn_residual)?;

        // Flatten back: [batch, context_window * embedding_size]
        let result = result.reshape((batch_size, self.config.input_size))?;

        Ok(result)
    }
}
