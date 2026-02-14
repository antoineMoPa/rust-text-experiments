use candle_core::{Tensor, D};
use candle_nn::{self as nn, VarBuilder};

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let weight = vb.get_with_hints(size, "weight", nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(size, "bias", nn::Init::Const(0.0))?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let x_norm = x_centered.broadcast_div(&std)?;
        x_norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}
