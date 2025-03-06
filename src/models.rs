pub trait RunStr {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error>;
}
