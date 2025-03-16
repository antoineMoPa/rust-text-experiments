use candle_nn::{init::DEFAULT_KAIMING_NORMAL, Linear};

pub fn custom_linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear, candle_core::Error> {
    let init_ws = DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 0.5;
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}
