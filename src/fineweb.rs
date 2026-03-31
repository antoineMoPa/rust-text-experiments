use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::{LargeStringArray, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const CACHE_DIR: &str = "fineweb_cache";
const HF_DATASET: &str = "HuggingFaceFW/fineweb";
const DEFAULT_SHARD: &str = "sample/10BT/000_00000.parquet";

/// Full pipeline: download shard if not cached, extract text if not cached.
/// Returns path to the extracted text file.
pub fn prepare_text(max_mb: usize, hf_token: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(CACHE_DIR)?;

    let text_cache = PathBuf::from(format!("{}/text_{}mb.txt", CACHE_DIR, max_mb));
    if text_cache.exists() {
        println!("Using cached text: {}", text_cache.display());
        return Ok(text_cache);
    }

    let parquet_cache = PathBuf::from(format!("{}/{}", CACHE_DIR, DEFAULT_SHARD));
    if !parquet_cache.exists() {
        download_shard(DEFAULT_SHARD, hf_token)?;
    } else {
        println!("Using cached shard: {}", parquet_cache.display());
    }

    println!("Extracting up to {}MB of text...", max_mb);
    let text = read_parquet_text(&parquet_cache, max_mb * 1024 * 1024)?;
    fs::write(&text_cache, &text)?;
    println!("Cached extracted text to {}", text_cache.display());

    Ok(text_cache)
}

fn download_shard(shard: &str, hf_token: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading shard {} from {}...", shard, HF_DATASET);
    let status = Command::new("hf")
        .args([
            "download",
            HF_DATASET,
            "--repo-type", "dataset",
            "--include", shard,
            "--local-dir", CACHE_DIR,
        ])
        .env("HF_TOKEN", hf_token)
        .status()
        .map_err(|e| if e.kind() == std::io::ErrorKind::NotFound {
            "hf not found — run: pip install huggingface_hub".into()
        } else {
            Box::from(e) as Box<dyn std::error::Error>
        })?;

    if !status.success() {
        return Err("hf download failed".into());
    }
    println!("Shard downloaded.");
    Ok(())
}

/// Read the `text` column from a Parquet file up to `max_bytes`.
pub fn read_parquet_text(path: &Path, max_bytes: usize) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let schema = builder.schema().clone();
    let text_col = schema
        .fields()
        .iter()
        .position(|f| f.name() == "text")
        .ok_or("No 'text' column found in parquet file")?;

    let col_type = schema.field(text_col).data_type().clone();
    if col_type != DataType::Utf8 && col_type != DataType::LargeUtf8 {
        return Err(format!("'text' column has unexpected type: {:?}", col_type).into());
    }

    let reader = builder.with_batch_size(1024).build()?;
    let mut out = String::with_capacity(max_bytes.min(64 * 1024 * 1024));

    'outer: for batch_result in reader {
        let batch = batch_result?;
        let col = batch.column(text_col);

        if col_type == DataType::Utf8 {
            let arr = col.as_any().downcast_ref::<StringArray>()
                .ok_or("Failed to downcast Utf8 column")?;
            for value in arr.iter() {
                if let Some(t) = value {
                    out.push_str(t);
                    out.push('\n');
                    if out.len() >= max_bytes { out.truncate(max_bytes); break 'outer; }
                }
            }
        } else {
            let arr = col.as_any().downcast_ref::<LargeStringArray>()
                .ok_or("Failed to downcast LargeUtf8 column")?;
            for value in arr.iter() {
                if let Some(t) = value {
                    out.push_str(t);
                    out.push('\n');
                    if out.len() >= max_bytes { out.truncate(max_bytes); break 'outer; }
                }
            }
        }
    }

    println!("Read {:.1} MB from {}", out.len() as f64 / 1_048_576.0, path.display());
    Ok(out)
}
