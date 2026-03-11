use candle_core::Tensor;
use candle_nn::{self as nn, Module};
use nn::VarBuilder;

#[cfg(all(not(target_os = "macos"), not(feature = "flash-attn")))]
use crate::flash_attn_op::flash_attn;

use crate::layer_norm::LayerNorm;

pub struct AttentionBlock {
    pub qkv_proj: nn::Linear,
    pub out_linear: nn::Linear,
    pub ffn_in: nn::Linear,
    pub ffn_out: nn::Linear,
    pub config: AttentionBlockConfig,
    #[cfg(target_os = "macos")]
    causal_mask: Tensor,
    pos_enc: Tensor,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

pub struct AttentionBlockConfig {
    pub num_attention_heads: usize,
    pub context_window: usize,
    pub embedding_size: usize,
    pub ffn_hidden: usize,
}

impl AttentionBlock {
    pub fn new(config: AttentionBlockConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        // Fused QKV: one projection embedding_size -> 3 * embedding_size
        let qkv_proj = nn::linear_b(
            config.embedding_size,
            3 * config.embedding_size,
            true,
            vb.pp("qkv_proj"),
        )?;

        let out_linear = nn::linear_b(
            config.embedding_size,
            config.embedding_size,
            false,
            vb.pp("out_linear"),
        )?;

        let ffn_in = nn::linear_b(
            config.embedding_size,
            config.ffn_hidden,
            true,
            vb.pp("ffn_in"),
        )?;
        let ffn_out = nn::linear_b(
            config.ffn_hidden,
            config.embedding_size,
            true,
            vb.pp("ffn_out"),
        )?;

        let device = vb.device();
        let seq_len = config.context_window;

        #[cfg(target_os = "macos")]
        let causal_mask = {
            let mask_data: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?
        };

        let mut pe_data: Vec<f32> = Vec::with_capacity(seq_len * config.embedding_size);
        for i in 0..seq_len {
            for j in 0..config.embedding_size {
                let val =
                    (i as f32) / (10000_f32).powf(2.0 * (j as f32) / config.embedding_size as f32);
                pe_data.push(if j % 2 == 0 { val.sin() } else { val.cos() });
            }
        }
        let pos_enc = Tensor::from_slice(&pe_data, (1, seq_len, config.embedding_size), device)?;

        let norm1 = LayerNorm::new(config.embedding_size, 1e-5, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(config.embedding_size, 1e-5, vb.pp("norm2"))?;

        Ok(Self {
            qkv_proj,
            out_linear,
            ffn_in,
            ffn_out,
            config,
            #[cfg(target_os = "macos")]
            causal_mask,
            pos_enc,
            norm1,
            norm2,
        })
    }

    pub fn position_encoding(&self) -> &Tensor {
        &self.pos_enc
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> Result<Tensor, candle_core::Error> {
        let batch_size = input.dim(0)?;
        let num_heads = self.config.num_attention_heads;
        let emb = self.config.embedding_size;
        let seq = self.config.context_window;
        let d_head = emb / num_heads;
        let scale = 1.0 / (d_head as f64).sqrt();

        let input = if train {
            nn::ops::dropout(input, 0.1)?
        } else {
            input.clone()
        };

        // Pre-norm before attention
        let normed = self.norm1.forward(&input)?;

        // Fused QKV projection: [batch, seq, 3*emb]
        let qkv = self.qkv_proj.forward(&normed)?;

        // Split into Q, K, V: each [batch, seq, emb]
        let q = qkv.narrow(2, 0, emb)?;
        let k = qkv.narrow(2, emb, emb)?;
        let v = qkv.narrow(2, emb * 2, emb)?;

        // Attention: [batch, seq, emb] -> [batch, seq, emb]
        #[cfg(all(not(target_os = "macos"), feature = "flash-attn"))]
        let result = {
            // candle-flash-attn expects [batch, seq, heads, d_head] (seq-major).
            let q = q.reshape((batch_size, seq, num_heads, d_head))?.contiguous()?;
            let k = k.reshape((batch_size, seq, num_heads, d_head))?.contiguous()?;
            let v = v.reshape((batch_size, seq, num_heads, d_head))?.contiguous()?;
            candle_flash_attn::flash_attn(&q, &k, &v, scale as f32, true)?
                .reshape((batch_size, seq, emb))?
        };

        #[cfg(all(not(target_os = "macos"), not(feature = "flash-attn")))]
        let result = {
            // Our homebrew flash_attn expects [batch, heads, seq, d_head] f32 (heads-first).
            let q = q
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            flash_attn(&q, &k, &v, scale as f32, true)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch_size, seq, emb))?
        };

        #[cfg(target_os = "macos")]
        let result = {
            // Standard attention: reshape to [batch, num_heads, seq, d_head]
            let q = q
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((batch_size, seq, num_heads, d_head))?
                .transpose(1, 2)?
                .contiguous()?;
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
            let scores = scores.broadcast_add(&self.causal_mask)?;
            let attn_weights = nn::ops::softmax(&scores, candle_core::D::Minus1)?;
            attn_weights
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch_size, seq, emb))?
        };

        // Output projection: [batch, seq, emb]
        let result = self.out_linear.forward(&result)?;

        // Residual connection
        let result = (result + &input)?;

        // Pre-norm before FFN
        let ffn_residual = result.clone();
        let normed2 = self.norm2.forward(&result)?;
        let result = self.ffn_in.forward(&normed2)?.gelu()?;
        let result = self.ffn_out.forward(&result)?;
        let result = (result + ffn_residual)?;

        Ok(result)
    }
}
