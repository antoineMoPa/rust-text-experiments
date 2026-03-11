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

use crate::hf::{load_env, require};

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
                "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04".to_string()
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
// Startup script
// ---------------------------------------------------------------------------

fn make_startup_script() -> &'static str {
    r#"#!/bin/bash
set -euo pipefail

echo "=== RunPod job starting ==="

apt-get update -qq && apt-get install -y time 2>&1 | tail -1

# Clone source at the exact branch + commit we were sent from
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

# Install Rust if not present
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "=== Build ==="
RUSTFLAGS="-C linker=gcc" cargo build --release --features flash-attn 2>&1 | tee /tmp/build.log

echo "=== Train ==="
make clean train-flash 2>&1 | tee /tmp/train.log

echo "=== Test ==="
make test_model 2>&1 | tee /tmp/test.log

echo "=== Results ==="
make results 2>&1 | tee data/results.txt

echo "=== Upload to HuggingFace ==="
pip install -q huggingface_hub
hf upload "$HF_REPO" ./data/ . --repo-type model

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
  list                               List all running pods
  send [--machine-type <GPU_TYPE>]   Create pod and start job
  status [<job_id>]                  Show pod status
  stop   [<job_id|pod_id>]           Delete the pod (accepts raw pod ID too)
  fetch  [<job_id>]                  Download trained model data from HuggingFace

If <job_id> is omitted, the most recent job in runpod_jobs/ is used.

Required env vars in .env:
  RUNPOD_API_KEY      RunPod API key
  GIT_REPO_URL        Git repo URL (e.g. https://github.com/you/rust-text-experiments)
  HF_TOKEN            HuggingFace token with write access
  HF_REPO             HuggingFace repo ID (e.g. you/rust-text-model)
Optional:
  RUNPOD_DOCKER_IMAGE Docker image (default: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04)

Examples:
  cargo run --release -- runpod send --machine-type NVIDIA_A40
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

pub fn send_job(machine_type: &str) -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let git_repo_url = require(&env, "GIT_REPO_URL")?;
    let hf_token = require(&env, "HF_TOKEN")?;
    let hf_repo = require(&env, "HF_REPO")?;
    let job_id = make_job_id();

    let (git_branch, git_commit) = check_git_pushed()?;
    println!("Branch {} at {} is pushed.", git_branch, &git_commit[..12]);
    println!("Starting job {} on {}...", job_id, machine_type);

    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(make_startup_script().as_bytes());

    let env_vars: Vec<(&str, String)> = vec![
        ("JOB_ID", job_id.clone()),
        ("STARTUP_B64", startup_b64),
        ("GIT_REPO_URL", git_repo_url),
        ("GIT_BRANCH", git_branch),
        ("GIT_COMMIT", git_commit),
        ("HF_TOKEN", hf_token),
        ("HF_REPO", hf_repo.clone()),
    ];

    let pod_id = if let Some(existing) = env.get("MACHINE_ID") {
        println!("Using existing pod {} (MACHINE_ID set).", existing);
        existing.clone()
    } else {
        println!("Creating pod...");
        let id = runpod.create_pod(&job_id, machine_type, env_vars, &config.docker_image)?;
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

pub fn test_hf_upload() -> Result<(), Box<dyn Error>> {
    let env = load_env();
    let config = RunpodConfig::from_env(&env)?;
    let hf_token = require(&env, "HF_TOKEN")?;
    let hf_repo = require(&env, "HF_REPO")?;

    let test_script = r#"#!/bin/bash
set -euo pipefail
echo "=== HF upload test ==="
pip install -q huggingface_hub
MARKER="runpod-hf-test-$(date +%s)"
echo "$MARKER" > /tmp/hf_test.txt
hf upload "$HF_REPO" /tmp/hf_test.txt hf_test.txt --repo-type model
echo "MARKER=$MARKER"
echo "=== Upload complete ==="
"#;

    let startup_b64 =
        base64::engine::general_purpose::STANDARD.encode(test_script.as_bytes());

    let env_vars: Vec<(&str, String)> = vec![
        ("STARTUP_B64", startup_b64),
        ("HF_TOKEN", hf_token.clone()),
        ("HF_REPO", hf_repo.clone()),
    ];

    let http = reqwest::blocking::Client::new();
    let runpod = RunpodClient::new(&http, &config.api_key);

    println!("Creating test pod...");
    let pod_id = runpod.create_pod("hf-test", "NVIDIA GeForce RTX 3090", env_vars, &config.docker_image)?;
    println!("Pod created: {}. Waiting for it to run and finish...", pod_id);
    runpod.wait_for_running(&pod_id)?;

    // Wait for the container to exit (pod goes EXITED or STOPPED)
    println!("Pod is running the test script...");
    loop {
        thread::sleep(Duration::from_secs(10));
        let pod = runpod.get_pod(&pod_id)?;
        let status = pod["desiredStatus"].as_str().unwrap_or("UNKNOWN");
        println!("  pod status: {}", status);
        match status {
            "EXITED" | "STOPPED" | "TERMINATED" => break,
            "FAILED" => {
                let _ = runpod.delete_pod(&pod_id);
                return Err("Test pod failed".into());
            }
            _ => {}
        }
    }

    println!("Pod finished. Cleaning up...");
    let _ = runpod.delete_pod(&pod_id);

    // Verify by downloading the marker file
    println!("Verifying: downloading hf_test.txt from {}...", hf_repo);
    let tmp = std::env::temp_dir().join("hf_test.txt");
    let status = Command::new("hf")
        .args(["download", &hf_repo, "hf_test.txt",
               "--local-dir", std::env::temp_dir().to_str().unwrap(),
               "--repo-type", "model"])
        .env("HF_TOKEN", &hf_token)
        .status()?;

    if !status.success() || !tmp.exists() {
        return Err("Could not download hf_test.txt — upload may have failed".into());
    }

    let contents = fs::read_to_string(&tmp)?;
    println!("hf_test.txt contents: {}", contents.trim());
    let _ = fs::remove_file(&tmp);

    println!("HuggingFace remote upload test: PASSED");
    Ok(())
}

pub fn fetch_job(job_id_opt: Option<&str>) -> Result<(), Box<dyn Error>> {
    let state = load_job(job_id_opt)?;
    let env = load_env();
    let hf_token = require(&env, "HF_TOKEN")?;
    crate::hf::download(&state.hf_repo, &hf_token)
}
