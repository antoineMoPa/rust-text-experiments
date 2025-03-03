use candle_core::Device;

pub trait RunStr {
    fn run_str(&self, input: &str, len: usize) -> Result<String, candle_core::Error>;
}

pub trait GetDevice {
    fn get_device(&self) -> Device;
}
