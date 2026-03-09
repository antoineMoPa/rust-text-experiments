use base64::Engine;
use chrono::Utc;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Env loading
// ---------------------------------------------------------------------------

fn load_env() -> HashMap<String, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let path = format!("{}/.env", home);
    let mut map = HashMap::new();
    let Ok(content) = fs::read_to_string(&path) else {
        return map;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    map
}

fn require(env: &HashMap<String, String>, key: &str) -> Result<String, Box<dyn Error>> {
    env.get(key)
        .cloned()
        .ok_or_else(|| format!("Missing required env var: {} (add to ~/.env)", key).into())
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct RunpodConfig {
    api_key: String,
    s3_access_key: String,
    s3_secret: String,
    s3_endpoint: String,
    s3_region: String,
    /// Network volume ID — also used as S3 bucket name and attached to pod.
    volume_id: String,
    docker_image: String,
}

impl RunpodConfig {
    fn from_env(env: &HashMap<String, String>) -> Result<Self, Box<dyn Error>> {
        let docker_image = env
            .get("RUNPOD_DOCKER_IMAGE")
            .cloned()
            .unwrap_or_else(|| {
                "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04".to_string()
            });
        Ok(Self {
            api_key: require(env, "RUNPOD_API_KEY")?,
            s3_access_key: require(env, "RUNPOD_S3_ACCESS_KEY")?,
            s3_secret: require(env, "RUNPOD_S3_SECRET")?,
            s3_endpoint: require(env, "RUNPOD_S3_ENDPOINT")?,
            s3_region: require(env, "RUNPOD_S3_REGION")?,
            volume_id: require(env, "RUNPOD_S3_VOLUME_ID")?,
            docker_image,
        })
    }
}

// ---------------------------------------------------------------------------
// Job ID
// ---------------------------------------------------------------------------

fn make_job_id() -> String {
    let now = Utc::now().format("%Y%m%d-%H%M%S");
    let uid = &Uuid::new_v4().to_string()[..7];
    format!("{}-{}", now, uid)
}

// ---------------------------------------------------------------------------
// Retry helper
// ---------------------------------------------------------------------------

fn with_retry<T, F>(label: &str, mut f: F) -> Result<T, Box<dyn Error>>
where
    F: FnMut() -> Result<T, Box<dyn Error>>,
{
    let delays = [2u64, 4, 8, 16, 32];
    for (attempt, &delay) in delays.iter().enumerate() {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                eprintln!("[retry] {} failed (attempt {}): {}", label, attempt + 1, e);
                thread::sleep(Duration::from_secs(delay));
            }
        }
    }
    f()
}

// ---------------------------------------------------------------------------
// AWS Signature V4 helpers
// ---------------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    to_hex(&h.finalize())
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

fn signing_key(secret: &str, date_str: &str, region: &str, service: &str) -> Vec<u8> {
    let k = format!("AWS4{}", secret);
    let k_date = hmac_sha256(k.as_bytes(), date_str.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn s3_auth_header(
    access_key: &str,
    secret: &str,
    region: &str,
    host: &str,
    uri_path: &str,
    body: &[u8],
    amz_date: &str,
    date_str: &str,
) -> String {
    let payload_hash = sha256_hex(body);
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";
    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, payload_hash, amz_date
    );
    let canonical_request = format!(
        "PUT\n{}\n\n{}\n{}\n{}",
        uri_path, canonical_headers, signed_headers, payload_hash
    );
    let credential_scope = format!("{}/{}/s3/aws4_request", date_str, region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );
    let key = signing_key(secret, date_str, region, "s3");
    let signature = to_hex(&hmac_sha256(&key, string_to_sign.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
        access_key, credential_scope, signed_headers, signature
    )
}

// ---------------------------------------------------------------------------
// S3 operations (pure HTTP, no AWS SDK)
// ---------------------------------------------------------------------------

struct S3Ops<'a> {
    client: &'a reqwest::blocking::Client,
    config: &'a RunpodConfig,
    host: String,
    base_url: String,
}

impl<'a> S3Ops<'a> {
    fn new(client: &'a reqwest::blocking::Client, config: &'a RunpodConfig) -> Self {
        let base_url = config.s3_endpoint.trim_end_matches('/').to_string();
        let host = base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap_or(&base_url)
            .to_string();
        Self { client, config, host, base_url }
    }

    fn put(&self, key: &str, body: Vec<u8>) -> Result<(), Box<dyn Error>> {
        let now = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_str = now.format("%Y%m%d").to_string();
        let uri_path = format!("/{}/{}", self.config.volume_id, key);
        let url = format!("{}{}", self.base_url, uri_path);
        let auth = s3_auth_header(
            &self.config.s3_access_key,
            &self.config.s3_secret,
            &self.config.s3_region,
            &self.host,
            &uri_path,
            &body,
            &amz_date,
            &date_str,
        );
        let payload_hash = sha256_hex(&body);
        let resp = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-amz-date", &amz_date)
            .header("x-amz-content-sha256", &payload_hash)
            .header("host", &self.host)
            .body(body)
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(format!("S3 PUT {} → HTTP {}: {}", key, status, text).into());
        }
        Ok(())
    }

    fn upload_file(&self, local_path: &str, key: &str) -> Result<(), Box<dyn Error>> {
        with_retry(&format!("upload:{}", key), || {
            let body = fs::read(local_path)?;
            self.put(key, body)
        })
    }

    fn upload_bytes(&self, bytes: Vec<u8>, key: &str) -> Result<(), Box<dyn Error>> {
        with_retry(&format!("upload:{}", key), || self.put(key, bytes.clone()))
    }
}

// ---------------------------------------------------------------------------
// Source tarball
// ---------------------------------------------------------------------------

/// Creates a .tar.gz of all source files needed to build on the pod.
/// Returns the path to the temp file.
fn create_source_tarball(job_id: &str) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let tarball = std::env::temp_dir().join(format!("{}-src.tar.gz", job_id));

    let mut paths: Vec<&str> = vec!["src/", "Cargo.toml"];
    if Path::new("Cargo.lock").exists() {
        paths.push("Cargo.lock");
    }
    if Path::new("build.rs").exists() {
        paths.push("build.rs");
    }

    let corpus = "smoll-generated-corpus/level_5/corpus.corpus";
    if Path::new(corpus).exists() {
        paths.push(corpus);
    } else {
        eprintln!("warning: corpus not found at {} — skipping", corpus);
    }

    for f in &[
        "data/model.dict",
        "data/model.bpe",
        "data/model.id",
        "data/model.safetensors",
    ] {
        if Path::new(f).exists() {
            paths.push(f);
        }
    }

    let status = Command::new("tar")
        .arg("czf")
        .arg(&tarball)
        .args(&paths)
        .status()?;

    if !status.success() {
        return Err("tar failed to create source tarball".into());
    }

    let size = fs::metadata(&tarball)?.len();
    println!(
        "Source tarball: {} ({:.1} MB)",
        tarball.display(),
        size as f64 / 1_048_576.0
    );

    Ok(tarball)
}

// ---------------------------------------------------------------------------
// RunPod REST client
// ---------------------------------------------------------------------------

struct RunpodClient<'a> {
    client: &'a reqwest::blocking::Client,
    api_key: &'a str,
}

impl<'a> RunpodClient<'a> {
    fn new(client: &'a reqwest::blocking::Client, api_key: &'a str) -> Self {
        Self { client, api_key }
    }

    fn create_pod(
        &self,
        name: &str,
        machine_type: &str,
        volume_id: &str,
        env_vars: Vec<(&str, String)>,
        docker_image: &str,
    ) -> Result<String, Box<dyn Error>> {
        let env_list: Vec<serde_json::Value> = env_vars
            .into_iter()
            .map(|(k, v)| serde_json::json!({ "key": k, "value": v }))
            .collect();

        let body = serde_json::json!({
            "name": name,
            "imageName": docker_image,
            "gpuTypeIds": [machine_type],
            "gpuCount": 1,
            "containerDiskInGb": 50,
            "networkVolumeId": volume_id,
            "env": env_list,
            "dockerArgs": "bash -c 'echo $STARTUP_B64 | base64 -d | bash'"
        });

        with_retry("create_pod", || {
            let resp = self
                .client
                .post("https://rest.runpod.io/v1/pods")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                return Err(format!("create_pod HTTP {}: {}", status, text).into());
            }

            let json: serde_json::Value = resp.json()?;
            let pod_id = json["id"]
                .as_str()
                .ok_or("create_pod: missing 'id' in response")?
                .to_string();
            Ok(pod_id)
        })
    }
}

// ---------------------------------------------------------------------------
// Startup script
// ---------------------------------------------------------------------------

fn make_startup_script() -> &'static str {
    r#"#!/bin/bash
set -euo pipefail

echo "=== RunPod job starting ==="

JOB_DIR="/runpod-volume/${JOB_ID}"

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Extract source to local disk (fast I/O for cargo build)
mkdir -p /workspace
tar xzf "${JOB_DIR}/src.tar.gz" -C /workspace
cd /workspace

echo "=== Build ==="
cargo build --release 2>&1 | tee "${JOB_DIR}/build.log"

echo "=== Train ==="
./target/release/rust-text-experiments train 2>&1 | tee "${JOB_DIR}/train.log"

echo "=== Test ==="
./target/release/rust-text-experiments test_all 2>&1 | tee "${JOB_DIR}/test.log"

# Copy trained model back to volume
cp -r data/ "${JOB_DIR}/data/"

# Write sentinel so the fetch command knows we're done
echo "done" > "${JOB_DIR}/done"

echo "=== Self-terminate ==="
POD_ID=$(cat "${JOB_DIR}/pod_id.txt")
curl -s -X DELETE "https://rest.runpod.io/v1/pods/${POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"

echo "=== Done ==="
"#
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn send_to_runpod(machine_type: &str) -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let job_id = make_job_id();

    println!("Starting job {} on {}...", job_id, machine_type);

    let http = reqwest::blocking::Client::new();
    let s3 = S3Ops::new(&http, &config);

    // --- Pack and upload source ---
    println!("Creating source tarball...");
    let tarball = create_source_tarball(&job_id)?;
    let tarball_key = format!("{}/src.tar.gz", job_id);
    println!("Uploading source tarball...");
    s3.upload_file(tarball.to_str().unwrap(), &tarball_key)?;
    let _ = fs::remove_file(&tarball); // clean up temp file

    // --- Create pod ---
    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(make_startup_script().as_bytes());

    let runpod = RunpodClient::new(&http, &config.api_key);
    let env_vars: Vec<(&str, String)> = vec![
        ("JOB_ID", job_id.clone()),
        ("RUNPOD_API_KEY", config.api_key.clone()),
        ("STARTUP_B64", startup_b64),
    ];

    println!("Creating pod...");
    let pod_id = runpod.create_pod(
        &job_id,
        machine_type,
        &config.volume_id,
        env_vars,
        &config.docker_image,
    )?;
    println!("Pod created: {}", pod_id);

    // Upload pod_id.txt to volume so the pod can self-terminate
    s3.upload_bytes(
        pod_id.as_bytes().to_vec(),
        &format!("{}/pod_id.txt", job_id),
    )?;

    // Save local job state
    fs::create_dir_all("runpod_jobs")?;
    let job_file = format!("runpod_jobs/{}.json", job_id);
    let state = serde_json::json!({
        "pod_id": pod_id,
        "machine_type": machine_type,
        "started_at": Utc::now().to_rfc3339(),
        "status": "running",
        "volume_job_dir": format!("{}", job_id),
    });
    fs::write(&job_file, serde_json::to_string_pretty(&state)?)?;

    println!(
        "\nJob {} launched on pod {}.\nResults will appear at /runpod-volume/{}/data/ on completion.\nJob state: {}",
        job_id, pod_id, job_id, job_file
    );
    Ok(())
}
