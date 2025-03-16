use candle_nn::Linear;

pub fn custom_linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear, candle_core::Error> {
    let init_ws = candle_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.12,
    };

    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;

    let init_bs = candle_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.12,
    };

    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}
