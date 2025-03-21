use candle_core::{Tensor, D};
use candle_nn::{self as nn, Module};
use nn::VarBuilder;

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
    pub output_size: usize,
}

impl AttentionBlock {
    pub fn new(config: AttentionBlockConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let mut linear: Vec<nn::Linear> = Vec::new();
        let mut qs: Vec<nn::Linear> = Vec::new();
        let mut ks: Vec<nn::Linear> = Vec::new();
        let mut vs: Vec<nn::Linear> = Vec::new();

        let s = config.input_size / config.num_attention_heads;

        for i in 0..config.num_attention_heads {
            linear.push(nn::linear_b(s, s, true, vb.pp(&format!("linear{}", i)))?);
            qs.push(nn::linear_b(s, s, true, vb.pp(&format!("q{}", i)))?);
            ks.push(nn::linear_b(s, s, true, vb.pp(&format!("k{}", i)))?);
            vs.push(nn::linear_b(s, s, true, vb.pp(&format!("v{}", i)))?);
        }

        let out_linear = nn::linear_b(
            config.input_size,
            config.output_size,
            false,
            vb.pp("out_linear")
        )?;

        Ok(Self { linear, qs, ks, vs, out_linear, config })
    }

    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, candle_core::Error> {
        let scale = 1.0 / ((self.config.input_size) as f64).sqrt();
        let result = q.matmul(&k.t()?)?;
        let result = (result * scale)?;
        let result = nn::ops::softmax(&result, D::Minus1)?;
        let result = result.matmul(&v)?;

        Ok(result)
    }

    pub fn position_encoding(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut position: Vec<f32> = Vec::new();

        for i in 0..self.config.context_window {
            for j in 0..self.config.embedding_size {
                let val = (i as f32) / (10000 as f32).powf(2.0 * (j as f32) / self.config.embedding_size as f32);
                if j % 2 == 0 {
                    position.push(val.sin());
                } else {
                    position.push(val.cos());
                }
            }
        }

        let position = Tensor::new(position, input.device())?;

        let batch_size = input.dim(0)?;
        let encoding = position.repeat(&[batch_size, 1])?;

        return Ok(encoding);
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let input = (self.position_encoding(input)? + input)?;
        let input = nn::ops::dropout(&input, 0.2)?;

        let mut results: Vec<Tensor> = Vec::new();

        for i in 0..self.config.num_attention_heads {
            let portion_size = self.config.embedding_size / self.config.num_attention_heads;
            // take a portion of every embedded token
            // input is a vector of size EMBEDDING_SIZE * CONTEXT_WINDOW with tokens next to each other, I want to take just a portion of each token
            let mut portions: Vec<Tensor> = Vec::new();

            for j in 0..self.config.context_window {
                let start = j * self.config.embedding_size;
                let end = start + portion_size;
                let indexes: Vec<u32> = ((start as u32)..(end as u32)).collect();
                let indexes = Tensor::new(indexes, input.device())?;
                let portion = input.index_select(&indexes, D::Minus1)?;
                portions.push(portion);
            }

            let portions = Tensor::cat(&portions, D::Minus1)?;
            let linear_output = portions.apply(&self.linear[i])?;

            let q = linear_output.apply(&self.qs[i])?;
            let k = linear_output.apply(&self.ks[i])?;
            let v = linear_output.apply(&self.vs[i])?;

            let result = self.scaled_dot_product_attention(&q, &k, &v)?;

            let result = (((result * 0.4)? + portions.clone())? * 0.15)?;

            results.push(result);
        }

        let result = Tensor::cat(&results, D::Minus1)?;
        let result = self.out_linear.forward(&result)?;

        return Ok(result);
    }
}
