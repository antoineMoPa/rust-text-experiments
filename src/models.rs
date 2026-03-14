pub trait RunStr {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error>;
}

pub trait PredictGreedy {
    fn predict_next_token_greedy(
        &self,
        input: &str,
        device: &candle_core::Device,
    ) -> Result<String, candle_core::Error>;
}
