use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::process::Command;

// ---------------------------------------------------------------------------
// Env loading (mirrors runpod.rs — shared if needed later)
// ---------------------------------------------------------------------------

pub fn load_env() -> HashMap<String, String> {
    // Start with actual environment variables
    let mut map: HashMap<String, String> = std::env::vars().collect();
    // .env file overrides (local dev)
    if let Ok(content) = fs::read_to_string(".env") {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                map.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    map
}

pub fn require(env: &HashMap<String, String>, key: &str) -> Result<String, Box<dyn Error>> {
    env.get(key)
        .cloned()
        .ok_or_else(|| format!("Missing required env var: {} (add to .env)", key).into())
}

// ---------------------------------------------------------------------------
// Core helpers (also called from runpod.rs)
// ---------------------------------------------------------------------------

fn run_hf_cli(args: &[&str], hf_token: &str) -> Result<(), Box<dyn Error>> {
    Command::new("hf")
        .args(args)
        .env("HF_TOKEN", hf_token)
        .status()
        .map_err(|e| if e.kind() == std::io::ErrorKind::NotFound {
            "hf not found — run: pip install huggingface_hub (installs the `hf` CLI)".into()
        } else {
            Box::from(e) as Box<dyn Error>
        })
        .and_then(|s| if s.success() { Ok(()) } else {
            Err(format!("hf {} failed", args[0]).into())
        })
}

pub fn upload(hf_repo: &str, hf_token: &str) -> Result<(), Box<dyn Error>> {
    println!("Uploading data/ to {}...", hf_repo);
    run_hf_cli(&["upload", hf_repo, "./data/", ".", "--repo-type", "model"], hf_token)?;
    println!("Uploaded data/ to {}.", hf_repo);
    Ok(())
}

pub fn upload_binary(hf_repo: &str, hf_token: &str) -> Result<(), Box<dyn Error>> {
    println!("=== Building binary ===");
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .map_err(|e| format!("Failed to run cargo: {}", e))?;
    if !status.success() {
        return Err("cargo build --release failed".into());
    }

    let binary = "target/release/rust-text-experiments";
    println!("Uploading binary to {}/bin/rust-text-experiments...", hf_repo);
    run_hf_cli(
        &["upload", hf_repo, binary, "bin/rust-text-experiments", "--repo-type", "model"],
        hf_token,
    )?;
    println!("Binary uploaded to {}/bin/rust-text-experiments.", hf_repo);
    Ok(())
}

pub fn download(hf_repo: &str, hf_token: &str) -> Result<(), Box<dyn Error>> {
    println!("Downloading data/ from {}...", hf_repo);
    run_hf_cli(&["download", hf_repo, "--local-dir", "data", "--repo-type", "model"], hf_token)?;
    println!("Downloaded data/ from {}.", hf_repo);
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI entry points
// ---------------------------------------------------------------------------

pub fn print_help() {
    println!(
        "Usage: cargo run --release -- hf <subcommand>

Subcommands:
  upload         Upload data/ to HuggingFace
  download       Download data/ from HuggingFace
  upload_binary  Upload compiled binary to HuggingFace (used by runpod send)

Required env vars in .env:
  HF_TOKEN  HuggingFace token with write access
  HF_REPO   HuggingFace repo ID (e.g. you/rust-text-model)"
    );
}

pub fn cmd_upload() -> Result<(), Box<dyn Error>> {
    let env = load_env();
    upload(&require(&env, "HF_REPO")?, &require(&env, "HF_TOKEN")?)
}

pub fn cmd_download() -> Result<(), Box<dyn Error>> {
    let env = load_env();
    download(&require(&env, "HF_REPO")?, &require(&env, "HF_TOKEN")?)
}

pub fn cmd_upload_binary() -> Result<(), Box<dyn Error>> {
    let env = load_env();
    upload_binary(&require(&env, "HF_REPO")?, &require(&env, "HF_TOKEN")?)
}
