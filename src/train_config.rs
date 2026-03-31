use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub embedding_size: usize,
    pub context_window: usize,
    pub num_attention_heads: usize,
    pub ffn_hidden: usize,
    pub num_blocks: usize,
    pub file_path: String,
    pub lr: f64,
    pub epochs: u32,
    pub token_batch_size: usize,
    pub use_bf16: bool,
    #[serde(default)]
    pub train_seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            embedding_size: 256,
            context_window: 128,
            num_attention_heads: 8,
            ffn_hidden: 512,
            num_blocks: 3,
            file_path: "smoll-generated-corpus/level_5/corpus.corpus".to_string(),
            lr: 7e-3,
            epochs: 1,
            token_batch_size: 32768,
            use_bf16: true,
            train_seed: 0,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.embedding_size % self.num_attention_heads != 0 {
            return Err(format!(
                "embedding_size={} must be divisible by num_attention_heads={}",
                self.embedding_size, self.num_attention_heads
            ));
        }
        let d_head = self.embedding_size / self.num_attention_heads;
        if d_head > 1024 {
            return Err(format!(
                "d_head={} (embedding_size={} / num_attention_heads={}) exceeds \
                 CUDA block size limit of 1024.",
                d_head, self.embedding_size, self.num_attention_heads
            ));
        }
        Ok(())
    }

    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            embedding_size: env_usize("EMBEDDING_SIZE", d.embedding_size),
            context_window: env_usize("CONTEXT_WINDOW", d.context_window),
            num_attention_heads: env_usize("NUM_ATTENTION_HEADS", d.num_attention_heads),
            ffn_hidden: env_usize("FFN_HIDDEN", d.ffn_hidden),
            num_blocks: env_usize("NUM_BLOCKS", d.num_blocks),
            file_path: std::env::var("FILE_PATH").unwrap_or(d.file_path),
            lr: env_f64("LR", d.lr),
            epochs: env_usize("EPOCHS", d.epochs as usize) as u32,
            token_batch_size: env_usize("TOKEN_BATCH_SIZE", d.token_batch_size),
            use_bf16: std::env::var("BF16").map(|v| v == "1").unwrap_or(true),
            train_seed: 0, // generated at training start if 0
        }
    }

    /// Print the config as env-var export lines (for display / scripts).
    pub fn to_env_lines(&self) -> Vec<String> {
        vec![
            format!("EMBEDDING_SIZE={}", self.embedding_size),
            format!("CONTEXT_WINDOW={}", self.context_window),
            format!("NUM_ATTENTION_HEADS={}", self.num_attention_heads),
            format!("FFN_HIDDEN={}", self.ffn_hidden),
            format!("NUM_BLOCKS={}", self.num_blocks),
            format!("FILE_PATH={}", self.file_path),
            format!("LR={}", self.lr),
            format!("EPOCHS={}", self.epochs),
            format!("TOKEN_BATCH_SIZE={}", self.token_batch_size),
            format!("BF16={}", if self.use_bf16 { "1" } else { "0" }),
        ]
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
