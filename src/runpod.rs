use base64::Engine;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::process::Command;
use std::thread;
use std::time::Duration;
use uuid::Uuid;

use crate::attention_predictor::TrainConfig;
use crate::hf::{load_env, require};

// ---------------------------------------------------------------------------
// Per-job train params (subset of TrainConfig that the caller may override)
// ---------------------------------------------------------------------------

pub struct SendParams {
    pub machine_type: String,
    pub config: TrainConfig,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct RunpodConfig {
    api_key: String,
    docker_image: String,
}

impl RunpodConfig {
    fn from_env(env: &HashMap<String, String>) -> Result<Self, Box<dyn Error>> {
        let docker_image = env
            .get("RUNPOD_DOCKER_IMAGE")
            .cloned()
            .unwrap_or_else(|| {
                "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04".to_string()
            });
        Ok(Self {
            api_key: require(env, "RUNPOD_API_KEY")?,
            docker_image,
        })
    }
}

// ---------------------------------------------------------------------------
// Job ID & state
// ---------------------------------------------------------------------------

fn make_job_id() -> String {
    let now = Utc::now().format("%Y%m%d-%H%M%S");
    let uid = &Uuid::new_v4().to_string()[..7];
    format!("{}-{}", now, uid)
}

#[derive(Debug, Serialize, Deserialize)]
struct JobState {
    job_id: String,
    pod_id: String,
    hf_repo: String,
    machine_type: String,
    started_at: String,
}

fn save_job(state: &JobState) -> Result<String, Box<dyn Error>> {
    fs::create_dir_all("runpod_jobs")?;
    let path = format!("runpod_jobs/{}.json", state.job_id);
    fs::write(&path, serde_json::to_string_pretty(state)?)?;
    Ok(path)
}

fn load_job(job_id_opt: Option<&str>) -> Result<JobState, Box<dyn Error>> {
    if let Some(id) = job_id_opt {
        let path = format!("runpod_jobs/{}.json", id);
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read {}: {}", path, e))?;
        return Ok(serde_json::from_str(&text)?);
    }

    // Find the most recently modified .json in runpod_jobs/
    let mut entries: Vec<(std::time::SystemTime, std::path::PathBuf)> =
        fs::read_dir("runpod_jobs")?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |x| x == "json"))
            .filter_map(|e| {
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((mtime, e.path()))
            })
            .collect();

    entries.sort_by(|a, b| b.0.cmp(&a.0)); // newest first

    let path = entries
        .into_iter()
        .next()
        .map(|(_, p)| p)
        .ok_or("No job files found in runpod_jobs/")?;

    let text = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&text)?)
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
        env_vars: Vec<(&str, String)>,
        docker_image: &str,
    ) -> Result<String, Box<dyn Error>> {
        let env_map: serde_json::Map<String, serde_json::Value> = env_vars
            .into_iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::String(v)))
            .collect();

        let body = serde_json::json!({
            "name": name,
            "imageName": docker_image,
            "gpuTypeIds": [machine_type],
            "gpuCount": 1,
            "containerDiskInGb": 50,
            "env": env_map,
            "dockerStartCmd": ["bash", "-c", "echo $STARTUP_B64 | base64 -d | bash"]
        });

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
    }

    fn list_pods(&self) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
        let resp = self
            .client
            .get("https://rest.runpod.io/v1/pods")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(format!("list_pods HTTP {}: {}", status, text).into());
        }

        let json: serde_json::Value = resp.json()?;
        Ok(json.as_array().cloned().unwrap_or_default())
    }

    fn get_pod(&self, pod_id: &str) -> Result<serde_json::Value, Box<dyn Error>> {
        with_retry("get_pod", || {
            let resp = self
                .client
                .get(format!("https://rest.runpod.io/v1/pods/{}", pod_id))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                return Err(format!("get_pod HTTP {}: {}", status, text).into());
            }
            Ok(resp.json()?)
        })
    }

    fn wait_for_running(&self, pod_id: &str) -> Result<(), Box<dyn Error>> {
        println!("Waiting for pod {} to be RUNNING...", pod_id);
        loop {
            let pod = self.get_pod(pod_id)?;
            let status = pod["desiredStatus"].as_str().unwrap_or("UNKNOWN");
            println!("  pod status: {}", status);
            match status {
                "RUNNING" => return Ok(()),
                "FAILED" | "TERMINATED" => {
                    return Err(format!("Pod {} entered status {}", pod_id, status).into());
                }
                _ => thread::sleep(Duration::from_secs(5)),
            }
        }
    }

    fn delete_pod(&self, pod_id: &str) -> Result<(), Box<dyn Error>> {
        with_retry("delete_pod", || {
            let resp = self
                .client
                .delete(format!("https://rest.runpod.io/v1/pods/{}", pod_id))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                return Err(format!("delete_pod HTTP {}: {}", status, text).into());
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Startup scripts
// ---------------------------------------------------------------------------

fn make_build_binary_script() -> &'static str {
    r#"#!/bin/bash
set -euo pipefail

shutdown_pod() {
    echo "=== Shutting down pod ==="
    for i in 1 2 3 4 5; do
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
            -H "Authorization: Bearer $RP_ADMIN_KEY")
        echo "delete HTTP $HTTP"
        [ "$HTTP" = "200" ] && break || true
        sleep $i
    done
}
trap shutdown_pod EXIT

on_error() {
    echo "=== FATAL ERROR at line $1 — stopping ==="
    exit 1
}
trap 'on_error $LINENO' ERR

echo "=== RunPod build_and_upload_binary starting ==="

apt-get update -qq && apt-get install -y curl git 2>&1 | tail -3

WORKDIR=$(mktemp -d)
git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$WORKDIR"
cd "$WORKDIR"

ACTUAL=$(git rev-parse HEAD)
if [ "$ACTUAL" != "$GIT_COMMIT" ]; then
    echo "ERROR: commit mismatch — expected $GIT_COMMIT, got $ACTUAL"
    exit 1
fi
echo "Commit verified: $GIT_COMMIT"

if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | awk '{if ($1+0 > 90) print 89; else print $1}')
export CUDA_COMPUTE_CAP
echo "Using CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP"

mkdir -p /workspace/.cargo /workspace/target
export CARGO_HOME=/workspace/.cargo
export CARGO_TARGET_DIR=/workspace/target
export CARGO_BUILD_JOBS=4

echo "=== Build ==="
cargo build --release 2>&1 | tee /tmp/build.log

echo "=== Upload binary to HuggingFace ==="
pip install -q huggingface_hub
hf upload "$HF_REPO" "$CARGO_TARGET_DIR/release/rust-text-experiments" bin/rust-text-experiments --repo-type model

echo "=== Done ==="
"#
}

fn make_startup_script() -> &'static str {
    r#"#!/bin/bash
set -euo pipefail

shutdown_pod() {
    echo "=== Shutting down pod ==="
    for i in 1 2 3 4 5; do
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
            -H "Authorization: Bearer $RP_ADMIN_KEY")
        echo "delete HTTP $HTTP"
        [ "$HTTP" = "200" ] && break || true
        sleep $i
    done
}
trap shutdown_pod EXIT

on_error() {
    echo "=== FATAL ERROR at line $1 — stopping ==="
    exit 1
}
trap 'on_error $LINENO' ERR

echo "=== RunPod job starting ==="

apt-get update -qq && apt-get install -y time 2>&1 | tail -1

# Clone source at the exact branch + commit for corpus generation
WORKDIR=$(mktemp -d)
git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$WORKDIR"
cd "$WORKDIR"
git submodule update --init --recursive
cd smoll-generated-corpus && bash make_corpus.sh && cd ..

# Verify we have the exact commit that was sent
ACTUAL=$(git rev-parse HEAD)
if [ "$ACTUAL" != "$GIT_COMMIT" ]; then
    echo "ERROR: commit mismatch — expected $GIT_COMMIT, got $ACTUAL"
    exit 1
fi
echo "Commit verified: $GIT_COMMIT"

# Download pre-built binary from HuggingFace (built locally with: cargo build --release)
pip install -q huggingface_hub
echo "=== Downloading binary from HuggingFace ==="
hf download "$HF_REPO" bin/rust-text-experiments --local-dir /tmp/bin --repo-type model
BIN=/tmp/bin/bin/rust-text-experiments
chmod +x "$BIN"
echo "Binary ready: $("$BIN" --version 2>/dev/null || echo ok)"

mkdir -p data

echo "=== Train ==="
"$BIN" train 2>&1 | tee /tmp/train.log

echo "=== Test ==="
"$BIN" test_all 2>&1 | tee /tmp/test.log

echo "=== Results ==="
"$BIN" print_results 2>&1 | tee data/results.txt

echo "=== Upload to HuggingFace ==="
for f in model.bpe model.config.json model.dict model.id model.safetensors; do
    if [ -f "data/$f" ]; then
        hf upload "$HF_REPO" "data/$f" "$f" --repo-type model
    else
        echo "Skipping $f (not found)"
    fi
done

echo "=== Done ==="
"#
}

// ---------------------------------------------------------------------------
// Public commands
// ---------------------------------------------------------------------------

pub fn print_help() {
    println!(
        "Usage: cargo run --release -- runpod <subcommand> [options]

Subcommands:
  list                                        List all running pods
  send [options]                              Create pod and start training job
  status [<job_id>]                           Show pod status
  stop   [<job_id|pod_id>] [--all]            Delete pod(s)
  fetch  [<job_id>]                           Download trained model data from HuggingFace
  build_and_upload_binary [--machine-type X]  Build binary on RunPod and upload to HuggingFace

send options (all optional, defaults come from env vars then built-in defaults):
  --machine-type <GPU>          GPU type (default: NVIDIA GeForce RTX 4090)
  --embedding-size <N>          Embedding size (default: 256)
  --context-window <N>          Context window (default: 128)
  --num-heads <N>               Attention heads (default: 8)
  --ffn-hidden <N>              FFN hidden size (default: 512)
  --num-blocks <N>              Transformer blocks (default: 2)
  --file-path <PATH>            Corpus file path (default: smoll-generated-corpus/level_5/corpus.corpus)
  --lr <F>                      Learning rate (default: 0.01)
  --warmup-batches <N>          LR warmup batches (default: 600)
  --epochs <N>                  Training epochs (default: 1)
  --batch-size <N>              Token batch size (default: 256)
  --micro-batch-size <N>        Micro batch size (default: 256)

If <job_id> is omitted for status/fetch, the most recent job in runpod_jobs/ is used.

Required env vars in .env:
  RUNPOD_API_KEY      RunPod API key
  GIT_REPO_URL        Git repo URL (e.g. https://github.com/you/rust-text-experiments)
  HF_TOKEN            HuggingFace token with write access
  HF_REPO             HuggingFace repo ID (e.g. you/rust-text-model)
Optional:
  RUNPOD_DOCKER_IMAGE Docker image (default: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)

Examples:
  cargo run --release -- runpod build_and_upload_binary --machine-type NVIDIA_A40
  cargo run --release -- runpod send --machine-type NVIDIA_A40
  cargo run --release -- runpod send --machine-type NVIDIA_A40 --embedding-size 256 --context-window 128 --num-heads 8 --ffn-hidden 512 --num-blocks 2 --lr 0.01 --epochs 1
  cargo run --release -- runpod status
  cargo run --release -- runpod fetch
  cargo run --release -- runpod stop"
    );
}

fn check_git_pushed() -> Result<(String, String), Box<dyn Error>> {
    let branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()?;
    if !branch.status.success() {
        return Err("Not in a git repository".into());
    }
    let branch = String::from_utf8(branch.stdout)?.trim().to_string();

    let local = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    let local = String::from_utf8(local.stdout)?.trim().to_string();

    let remote_ref = format!("origin/{}", branch);
    let upstream = Command::new("git")
        .args(["rev-parse", &remote_ref])
        .output()?;
    if !upstream.status.success() {
        return Err(format!(
            "Branch '{}' not found on origin. Push it first: git push origin {}",
            branch, branch
        ).into());
    }
    let upstream = String::from_utf8(upstream.stdout)?.trim().to_string();

    if local != upstream {
        return Err(format!(
            "Branch '{}' is not up to date with remote ({} vs {}). Push first.",
            branch, &local[..12], &upstream[..12]
        ).into());
    }

    Ok((branch, local))
}

pub fn send_job(params: SendParams) -> Result<(), Box<dyn Error>> {
    let machine_type = &params.machine_type;
    let env = load_env();
    let runpod_config = RunpodConfig::from_env(&env)?;
    let git_repo_url = require(&env, "GIT_REPO_URL")?;
    let hf_token = require(&env, "HF_TOKEN")?;
    let hf_repo = require(&env, "HF_REPO")?;
    let job_id = make_job_id();

    let (git_branch, git_commit) = check_git_pushed()?;
    println!("Branch {} at {} is pushed.", git_branch, &git_commit[..12]);
    println!("Starting job {} on {}...", job_id, machine_type);
    println!("Train config:");
    for line in params.config.to_env_lines() {
        println!("  {}", line);
    }

    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &runpod_config.api_key);

    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(make_startup_script().as_bytes());

    let mut env_vars: Vec<(&str, String)> = vec![
        ("JOB_ID", job_id.clone()),
        ("STARTUP_B64", startup_b64),
        ("GIT_REPO_URL", git_repo_url),
        ("GIT_BRANCH", git_branch),
        ("GIT_COMMIT", git_commit),
        ("HF_TOKEN", hf_token),
        ("HF_REPO", hf_repo.clone()),
        ("RP_ADMIN_KEY", runpod_config.api_key.clone()),
    ];

    // Append all TrainConfig fields as env vars so the pod binary picks them up
    for line in params.config.to_env_lines() {
        if let Some((k, v)) = line.split_once('=') {
            // We need 'static keys — use Box::leak for the small number of keys
            let k: &'static str = Box::leak(k.to_string().into_boxed_str());
            env_vars.push((k, v.to_string()));
        }
    }

    let pod_id = if let Some(existing) = env.get("MACHINE_ID") {
        println!("Using existing pod {} (MACHINE_ID set).", existing);
        existing.clone()
    } else {
        println!("Creating pod...");
        let id = runpod.create_pod(&job_id, machine_type, env_vars, &runpod_config.docker_image)?;
        println!("Pod created: {}", id);
        runpod.wait_for_running(&id)?;
        id
    };

    let state = JobState {
        job_id: job_id.clone(),
        pod_id: pod_id.clone(),
        hf_repo,
        machine_type: machine_type.to_string(),
        started_at: Utc::now().to_rfc3339(),
    };
    let job_file = save_job(&state)?;

    println!(
        "\nJob {} launched on pod {}.\n\
         Pod is cloning repo and will start building shortly.\n\
         Check progress:  cargo run --release -- runpod status\n\
         Fetch results:   cargo run --release -- runpod fetch\n\
         Stop pod:        cargo run --release -- runpod stop\n\
         Job state saved: {}",
        job_id, pod_id, job_file
    );
    Ok(())
}

pub fn status_job(job_id_opt: Option<&str>) -> Result<(), Box<dyn Error>> {
    let state = load_job(job_id_opt)?;
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    println!("Job ID:      {}", state.job_id);
    println!("Pod ID:      {}", state.pod_id);
    println!("HF repo:     {}", state.hf_repo);
    println!("Machine:     {}", state.machine_type);
    println!("Started at:  {}", state.started_at);

    match runpod.get_pod(&state.pod_id) {
        Ok(pod) => {
            let desired = pod["desiredStatus"].as_str().unwrap_or("unknown");
            let uptime = pod["runtime"]["uptimeInSeconds"]
                .as_u64()
                .map(|s| format!("{}s uptime", s))
                .unwrap_or_else(|| "-".to_string());
            println!("Pod status:  {} ({})", desired, uptime);
        }
        Err(e) => println!("Pod status:  (API error: {})", e),
    }

    Ok(())
}

pub fn list_pods() -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    let pods = runpod.list_pods()?;
    if pods.is_empty() {
        println!("No pods running.");
        return Ok(());
    }
    println!("{:<20} {:<12} {}", "POD ID", "STATUS", "NAME");
    for pod in &pods {
        let id = pod["id"].as_str().unwrap_or("-");
        let status = pod["desiredStatus"].as_str().unwrap_or("-");
        let name = pod["name"].as_str().unwrap_or("-");
        println!("{:<20} {:<12} {}", id, status, name);
    }
    Ok(())
}

pub fn stop_job(id_opt: Option<&str>, all: bool) -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    if all {
        let pods = runpod.list_pods()?;
        if pods.is_empty() {
            println!("No pods running.");
            return Ok(());
        }
        for pod in &pods {
            let pod_id = pod["id"].as_str().unwrap_or("");
            print!("Stopping pod {}... ", pod_id);
            match runpod.delete_pod(pod_id) {
                Ok(()) => println!("done."),
                Err(e) => println!("failed: {}", e),
            }
        }
        return Ok(());
    }

    let pod_id = match load_job(id_opt) {
        Ok(state) => {
            println!("Stopping pod {} (job {})...", state.pod_id, state.job_id);
            state.pod_id
        }
        Err(_) => match id_opt {
            Some(id) => {
                println!("Stopping pod {}...", id);
                id.to_string()
            }
            None => {
                eprintln!("No local job found. Use 'runpod list' to see running pods, then:");
                eprintln!("  cargo run --release -- runpod stop <pod_id>");
                eprintln!("  cargo run --release -- runpod stop --all");
                return Ok(());
            }
        },
    };

    runpod.delete_pod(&pod_id)?;
    println!("Pod {} deleted.", pod_id);
    Ok(())
}


pub fn test_shutdown(machine_type: &str) -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;

    // Minimal script: print env, attempt self-delete, log visible in RunPod dashboard only.
    let test_script = r#"#!/bin/bash
set -uo pipefail

echo "=== shutdown test starting at $(date -u) ==="

echo ""
echo "--- All RUNPOD_* env vars ---"
printenv | grep -i runpod || echo "(none found)"

echo ""
echo "--- POD_ID from RUNPOD_POD_ID: '${RUNPOD_POD_ID:-UNSET}' ---"
# Note: RunPod injects its own RUNPOD_API_KEY (pod-scoped, 403s on delete).
# We pass our admin key as RP_ADMIN_KEY to avoid the collision.
echo "--- RP_ADMIN_KEY set: $([ -n "${RP_ADMIN_KEY:-}" ] && echo YES || echo NO) ---"

echo ""
echo "Sleeping 5s before attempting self-delete..."
sleep 5

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "ERROR: RUNPOD_POD_ID is not set — cannot self-delete"
else
    echo ""
    echo "--- Attempting DELETE /v1/pods/$RUNPOD_POD_ID with RP_ADMIN_KEY ---"
    STATUS=$(curl -s -o /tmp/delete_resp.txt -w "%{http_code}" \
        -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
        -H "Authorization: Bearer $RP_ADMIN_KEY")
    BODY=$(cat /tmp/delete_resp.txt)
    echo "HTTP status: $STATUS"
    echo "Response body: $BODY"
fi

echo "=== done at $(date -u) ==="
"#;

    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(test_script.as_bytes());

    let env_vars: Vec<(&str, String)> = vec![
        ("STARTUP_B64", startup_b64),
        ("RP_ADMIN_KEY", config.api_key.clone()),
    ];

    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    println!("Creating shutdown-test pod...");
    let pod_id = runpod.create_pod(
        "shutdown-test",
        machine_type,
        env_vars,
        &config.docker_image,
    )?;
    println!("Pod created: {}", pod_id);
    println!("Check RunPod dashboard logs to see the HTTP status, and whether pod {} disappears.", pod_id);
    println!("Force-stop with:  cargo run --release -- runpod stop {}", pod_id);
    Ok(())
}

pub fn fetch_job(job_id_opt: Option<&str>) -> Result<(), Box<dyn Error>> {
    let state = load_job(job_id_opt)?;
    let env = load_env();
    let hf_token = require(&env, "HF_TOKEN")?;
    crate::hf::download(&state.hf_repo, &hf_token)
}

pub fn build_and_upload_binary(machine_type: &str) -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let git_repo_url = require(&env, "GIT_REPO_URL")?;
    let hf_token = require(&env, "HF_TOKEN")?;
    let hf_repo = require(&env, "HF_REPO")?;

    let (git_branch, git_commit) = check_git_pushed()?;
    println!("Branch {} at {} is pushed.", git_branch, &git_commit[..12]);
    println!("Starting build pod on {}...", machine_type);

    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(make_build_binary_script().as_bytes());

    let env_vars: Vec<(&str, String)> = vec![
        ("STARTUP_B64", startup_b64),
        ("GIT_REPO_URL", git_repo_url),
        ("GIT_BRANCH", git_branch),
        ("GIT_COMMIT", git_commit),
        ("HF_TOKEN", hf_token),
        ("HF_REPO", hf_repo.clone()),
        ("RP_ADMIN_KEY", config.api_key.clone()),
    ];

    let pod_id = runpod.create_pod("build-binary", machine_type, env_vars, &config.docker_image)?;
    println!("Pod created: {}. Waiting for it to start...", pod_id);
    runpod.wait_for_running(&pod_id)?;

    println!("Pod is building and uploading binary (this takes ~10 min)...");
    println!("Monitor logs in RunPod dashboard, pod: {}", pod_id);
    println!("When done the binary will be at {}/bin/rust-text-experiments", hf_repo);
    println!("Stop the pod manually with:  cargo run --release -- runpod stop {}", pod_id);

    Ok(())
}
