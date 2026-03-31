use attention_predictor::{create_model, get_pretrained_dict};
use candle_core::Var;

use crate::{
    attention_predictor::{get_device, load_vocab, Model},
    model_tests::{json_test, per_epoch_scores, print_results, qa_test, self_test, test_all},
    token_utils::STOP_TOKEN,
    train_config::TrainConfig,
};

mod attention_block;
mod attention_predictor;
#[cfg(not(target_os = "macos"))]
mod flash_attn_op;
mod grad_accum;
mod layer_norm;
mod model_tests;
mod models;
mod fineweb;
mod hf;
mod runpod;
mod token_utils;
mod train_config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let command = args.first().map(|s| s.as_str()).unwrap_or("");

    let device = get_device()?;

    if command == "print_stats" {
        println!("Loading test model");
        let model = Model::load_from_path("data/model", &device)?;
        model.print_stats()?;
        return Ok(());
    }

    if command == "param_count" {
        let model = load_vocab("data/model", &device)?;
        println!("\nParameter count (vocab_size={}):", model.dict.len());
        model.count_params();
        return Ok(());
    }

    if command == "train" {
        if args.iter().any(|a| a == "--help" || a == "-h") {
            println!("Usage: train [FLAGS]\n\nFlags:\n  --bf16          Use bfloat16 (falls back to f32 if GPU doesn't support it)\n  --new-epoch [N] Continue training saved model for N more epochs (default 1)\n  --resume        Resume from data/model.ckpt checkpoint");
            return Ok(());
        }
        let device = get_device()?;
        let mut config = TrainConfig::from_env();
        if args.iter().any(|a| a == "--bf16") {
            config.use_bf16 = true;
        }
        let lr = config.lr;
        let file_path = config.file_path.clone();
        let (dict, tokens, bpe) = get_pretrained_dict(&file_path)?;

        println!("Training on {} tokens", tokens.len());

        let new_epoch_flag = args.iter().position(|a| a == "--new-epoch");
        let resume_flag = args.iter().any(|a| a == "--resume");

        if resume_flag {
            let ckpt: serde_json::Value = std::fs::read_to_string("data/model.ckpt")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let resume_epoch = ckpt["epoch"].as_u64().unwrap_or(0) as u32;
            let resume_batch = ckpt["batch"].as_u64().unwrap_or(0) as usize;
            println!("Resuming from epoch={} batch={}", resume_epoch, resume_batch);
            let mut model = Model::load_from_path("data/model", &device)?;
            model.simple_train(tokens, &device, lr, Some((resume_epoch, resume_batch)))?;
            model.save_to_path("data/model");
        } else if let Some(pos) = new_epoch_flag {
            let n = args.get(pos + 1).and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            println!("Continuing training for {} more epoch(s)", n);
            let mut model = Model::load_from_path("data/model", &device)?;
            model.config.epochs = n;
            model.simple_train(tokens, &device, lr, None)?;
            model.save_to_path("data/model");
        } else {
            println!("Training new model");
            let mut model = create_model(&dict, bpe, &device, config)?;
            model.save_to_path("data/model");
            model.simple_train(tokens, &device, lr, None)?;
            model.save_to_path("data/model");
        }

        return Ok(());
    }

    if command == "fine-tune" || command == "fine-tune-lr-range" {
        use std::io::{BufRead, BufReader};
        let ft_path = std::env::var("FINETUNE_PATH")
            .unwrap_or_else(|_| "fine_tune/finetune_train.json".to_string());

        let mut model = Model::load_from_path("data/model", &device)?;

        // Convert NDJSON pairs to a flat token stream: "<prompt>\n<response_json>\n<stop>\n"
        let file = std::fs::File::open(&ft_path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot open {}: {}", ft_path, e)))?;
        let mut text = String::new();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if line.trim().is_empty() { continue; }
            let v: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let prompt = v["prompt"].as_str().unwrap_or("").to_string();
            let response = serde_json::to_string(&v["response"])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            text.push_str(&prompt);
            text.push('\n');
            text.push_str(&response);
            text.push('\n');
            text.push_str(token_utils::STOP_TOKEN);
            text.push('\n');
        }
        let token_strs = model.bpe.tokenize(&text);
        drop(text);
        let tokens: Vec<u32> = token_strs.iter().map(|t| model.token_to_id_pub(t)).collect();
        drop(token_strs);
        println!("Loaded {} tokens from {}", tokens.len(), ft_path);

        if command == "fine-tune-lr-range" {
            let lr_lo: f64 = std::env::var("LR_LO").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-6);
            let lr_hi: f64 = std::env::var("LR_HI").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-2);
            let max_batches: usize = std::env::var("MAX_BATCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
            println!("LR range test on fine-tune data: {:.1e} → {:.1e} over {} batches", lr_lo, lr_hi, max_batches);
            model.lr_range_test(tokens, &device, lr_lo, lr_hi, max_batches)?;
        } else {
            let epochs: u32 = std::env::var("EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

            // Auto LR range test — reload a fresh copy so weights are untouched for fine-tuning
            let lr_lo: f64 = std::env::var("LR_LO").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-6);
            let lr_hi: f64 = std::env::var("LR_HI").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-2);
            let max_batches: usize = std::env::var("MAX_BATCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
            println!("Auto LR range test: {:.1e} → {:.1e} over {} batches", lr_lo, lr_hi, max_batches);
            let mut probe = Model::load_from_path("data/model", &device)?;
            let best_lr = probe.lr_range_test(tokens.clone(), &device, lr_lo, lr_hi, max_batches)?;
            let lr = best_lr.unwrap_or(1e-4);
            println!("Using LR = {:.2e} for fine-tuning", lr);

            model.config.epochs = epochs;
            println!("Fine-tuning on {} tokens for {} epochs at lr={:.2e}", tokens.len(), epochs, lr);
            model.simple_train(tokens, &device, lr, None)?;
            model.save_to_path("data/model");
            println!("Fine-tuned model saved to data/model");
        }
        return Ok(());
    }

    if command == "merge" {
        let path_a = "data/model_a";
        let path_b = "data/model_b";
        let path_average = "data/model_a_b_average";
        println!("Merging models {} {}", path_a, path_b);

        let device = get_device()?;
        let model_a = Model::load_from_path(path_a, &device)?;
        let model_b = Model::load_from_path(path_b, &device)?;
        let model_average = Model::load_from_path(path_a, &device)?;

        let data_a = model_a.var_map.data().lock().unwrap();
        let data_b = model_b.var_map.data().lock().unwrap();
        let mut data_average = model_average.var_map.data().lock().unwrap();

        data_a.keys().for_each(|key| {
            let value_a = data_a.get(key).unwrap().as_tensor();
            let value_b = data_b.get(key).unwrap().as_tensor();
            let average = ((value_a + value_b).unwrap() / 2.0).unwrap();
            data_average
                .insert(key.clone(), Var::from_tensor(&average).unwrap())
                .unwrap();
            println!("merged {}", key);
        });

        // Release lock so we can save!
        drop(data_average);

        model_average.save_to_path(path_average);

        println!("Merged both models to {}", path_average);

        return Ok(());
    }

    if command == "run" {
        let mut input = args[1..].join(" ") + " ";
        println!("Completing: '{:?}'", input);
        println!("Loading model");
        let model = Model::load_from_path("data/model", &device)?;
        loop {
            let pred = model.predict_next_token(input.as_str(), &device)?;
            if pred == STOP_TOKEN {
                println!();
                break;
            }
            input = input + pred.as_str();
            print!("{}", pred);
        }
        return Ok(());
    }

    if command == "tokenize" {
        let input = args[1..].join(" ");
        let bpe = crate::token_utils::Bpe::load("data/model.bpe")
            .unwrap_or_else(|_| crate::token_utils::Bpe::new_empty());
        let tokens = bpe.tokenize(&input);
        println!("{:?}", tokens);
        return Ok(());
    }

    if command == "self_test" {
        self_test()?;
        return Ok(());
    }

    if command == "qa_test" {
        qa_test()?;
        return Ok(());
    }

    if command == "json_test" {
        json_test()?;
        return Ok(());
    }

    if command == "test_all" {
        test_all()?;
        return Ok(());
    }

    if command == "print_results" {
        print_results()?;
        return Ok(());
    }

    if command == "sweep-lr" {
        let config = TrainConfig::from_env();
        let file_path = config.file_path.clone();
        let (dict, tokens, bpe) = get_pretrained_dict(&file_path)?;

        // 12 log-spaced LRs from 3e-4 to 1e-2
        let n = 12usize;
        let lr_lo = 3e-4f64;
        let lr_hi = 1e-2f64;
        let lrs: Vec<f64> = (0..n)
            .map(|i| (lr_lo.ln() + i as f64 / (n - 1) as f64 * (lr_hi.ln() - lr_lo.ln())).exp())
            .collect();

        for lr in lrs {
            println!("=== sweep-lr: LR = {:.2e} ===", lr);
            let mut model = create_model(&dict, bpe.clone(), &device, config.clone())?;
            model.simple_train(tokens.clone(), &device, lr, None)?;

            match per_epoch_scores(&model, &device) {
                Ok((l2, qa, json)) => {
                    let entry = serde_json::json!({
                        "Model_ID": model.model_id,
                        "LR": lr,
                        "Self_Test_Score_L2": l2,
                        "QA_Test_Score": qa,
                        "JSON_Test_Score": json,
                    });
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("lr_sweep.log")
                    {
                        use std::io::Write as W;
                        let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap());
                    }
                    println!("sweep result: LR={:.2e} L2={:.3} QA={:.3} JSON={:.3}", lr, l2, qa, json);
                }
                Err(e) => eprintln!("sweep score failed for LR={:.2e}: {}", lr, e),
            }
        }

        return Ok(());
    }

    if command == "lr-range-test" {
        let config = TrainConfig::from_env();
        let file_path = config.file_path.clone();
        let (_, tokens, _) = get_pretrained_dict(&file_path)?;
        let mut model = Model::load_from_path("data/model", &device)?;

        let lr_lo: f64 = std::env::var("LR_LO").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-5);
        let lr_hi: f64 = std::env::var("LR_HI").ok().and_then(|v| v.parse().ok()).unwrap_or(5e-2);
        let max_batches: usize = std::env::var("MAX_BATCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(500);

        println!("LR range test: {:.1e} → {:.1e} over {} batches", lr_lo, lr_hi, max_batches);
        model.lr_range_test(tokens, &device, lr_lo, lr_hi, max_batches)?;

        return Ok(());
    }

    if command == "sweep-corpus" {
        let config = TrainConfig::from_env();
        let lr = config.lr;
        let file_path = config.file_path.clone();
        let (dict, tokens, bpe) = get_pretrained_dict(&file_path)?;

        // 10 evenly-spaced rates: 100%, 90%, ..., 10%
        let rates: Vec<f64> = (1..=10).rev().map(|i| i as f64 * 0.1).collect();

        for rate in rates {
            let n = ((tokens.len() as f64) * rate).round() as usize;
            let subset = tokens[..n].to_vec();
            println!(
                "=== sweep-corpus: rate={:.0}% tokens={} ===",
                rate * 100.0,
                n
            );
            let mut model = create_model(&dict, bpe.clone(), &device, config.clone())?;
            model.simple_train(subset, &device, lr, None)?;

            match per_epoch_scores(&model, &device) {
                Ok((l2, qa, json)) => {
                    let entry = serde_json::json!({
                        "Model_ID": model.model_id,
                        "Corpus_Rate": rate,
                        "Num_Tokens": n,
                        "Self_Test_Score_L2": l2,
                        "QA_Test_Score": qa,
                        "JSON_Test_Score": json,
                    });
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("corpus_sweep.log")
                    {
                        use std::io::Write as W;
                        let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap());
                    }
                    println!(
                        "sweep result: rate={:.0}% tokens={} L2={:.3} QA={:.3} JSON={:.3}",
                        rate * 100.0, n, l2, qa, json
                    );
                }
                Err(e) => eprintln!("sweep score failed for rate={:.0}%: {}", rate * 100.0, e),
            }
        }

        return Ok(());
    }

    if command == "train_fineweb" {
        let max_mb: usize = args.iter().find(|a| !a.starts_with('-')).and_then(|v| v.parse().ok()).unwrap_or(128);
        let resume_flag = args.iter().any(|a| a == "--resume");
        let env = hf::load_env();
        let hf_token = hf::require(&env, "HF_TOKEN")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Download shard + extract text (both steps cached)
        let text_path = fineweb::prepare_text(max_mb, &hf_token)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut config = TrainConfig::from_env();
        config.file_path = text_path.to_string_lossy().to_string();
        let lr = config.lr;

        if resume_flag {
            let ckpt: serde_json::Value = std::fs::read_to_string("data/model.ckpt")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            let resume_epoch = ckpt["epoch"].as_u64().unwrap_or(0) as u32;
            let resume_batch = ckpt["batch"].as_u64().unwrap_or(0) as usize;
            println!("Resuming from epoch={} batch={}", resume_epoch, resume_batch);
            // BPE vocab must be re-derived the same way; tokens are rebuilt from cached text.
            let (_, tokens, _) = attention_predictor::get_pretrained_dict_sampled(&config.file_path, 10 * 1024 * 1024)?;
            let mut model = Model::load_from_path("data/model", &device)?;
            model.simple_train(tokens, &device, lr, Some((resume_epoch, resume_batch)))?;
            model.save_to_path("data/model");
        } else {
            // Learn BPE from a 10MB sample to avoid OOM on large corpora
            let (dict, tokens, bpe) = attention_predictor::get_pretrained_dict_sampled(&config.file_path, 10 * 1024 * 1024)?;
            println!("Training on {} tokens from FineWeb ({}MB)", tokens.len(), max_mb);
            let mut model = create_model(&dict, bpe, &device, config)?;
            model.save_to_path("data/model");
            model.simple_train(tokens, &device, lr, None)?;
            model.save_to_path("data/model");
        }
        return Ok(());
    }

    if command == "hf" {
        let sub = args.get(1).map(|s| s.as_str()).unwrap_or("help");
        match sub {
            "upload" => hf::cmd_upload()?,
            "download" => hf::cmd_download()?,
            "upload_binary" => hf::cmd_upload_binary()?,
            _ => hf::print_help(),
        }
        return Ok(());
    }

    if command == "runpod" {
        if args.iter().any(|a| a == "--help" || a == "-h") {
            runpod::print_help();
            return Ok(());
        }
        let sub = args.get(1).map(|s| s.as_str()).unwrap_or("help");
        match sub {
            "help" => runpod::print_help(),
            "send" => {
                fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
                    args.iter()
                        .position(|a| a == flag)
                        .and_then(|i| args.get(i + 1))
                        .map(|s| s.as_str())
                }
                fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
                    flag_str(args, flag).and_then(|v| v.parse().ok())
                }
                fn flag_f64(args: &[String], flag: &str) -> Option<f64> {
                    flag_str(args, flag).and_then(|v| v.parse().ok())
                }
                fn flag_u32(args: &[String], flag: &str) -> Option<u32> {
                    flag_str(args, flag).and_then(|v| v.parse().ok())
                }

                let machine_type = flag_str(&args, "--machine-type")
                    .unwrap_or("NVIDIA GeForce RTX 4090")
                    .to_string();

                // Build config: start from env defaults, then override with CLI flags
                let mut config = TrainConfig::from_env();
                if let Some(v) = flag_usize(&args, "--embedding-size")    { config.embedding_size = v; }
                if let Some(v) = flag_usize(&args, "--context-window")    { config.context_window = v; }
                if let Some(v) = flag_usize(&args, "--num-heads")         { config.num_attention_heads = v; }
                if let Some(v) = flag_usize(&args, "--ffn-hidden")        { config.ffn_hidden = v; }
                if let Some(v) = flag_usize(&args, "--num-blocks")        { config.num_blocks = v; }
                if let Some(v) = flag_str(&args, "--file-path")           { config.file_path = v.to_string(); }
                if let Some(v) = flag_f64(&args, "--lr")                  { config.lr = v; }
                if let Some(v) = flag_u32(&args, "--epochs")              { config.epochs = v; }
                if let Some(v) = flag_usize(&args, "--batch-size")        { config.token_batch_size = v; }
                if args.iter().any(|a| a == "--bf16") { config.use_bf16 = true; }

                let no_shutdown = args.iter().any(|a| a == "--no-shutdown");
                let continue_training = args.iter().any(|a| a == "--continue");
                runpod::send_job(runpod::SendParams { machine_type, config, no_shutdown, continue_training })?;
            }
            "list" => runpod::list_pods()?,
            "status" => runpod::status_job(args.get(2).map(|s| s.as_str()))?,
            "stop" => {
                let all = args.iter().any(|a| a == "--all");
                let id = if all { None } else { args.get(2).map(|s| s.as_str()) };
                runpod::stop_job(id, all)?;
            }
            "fetch" => runpod::fetch_job(args.get(2).map(|s| s.as_str()))?,
            "build_and_upload_binary" => {
                let machine_type = args
                    .iter()
                    .position(|a| a == "--machine-type")
                    .and_then(|i| args.get(i + 1))
                    .map(|s| s.as_str())
                    .unwrap_or("NVIDIA GeForce RTX 4090");
                let no_shutdown = args.iter().any(|a| a == "--no-shutdown");
                runpod::build_and_upload_binary(machine_type, no_shutdown)?;
            }
            other => {
                eprintln!("Unknown runpod subcommand: '{}'. Run 'runpod help' for usage.", other);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    println!("Usage: rust-text-experiments <command>\nCommands: train, train_fineweb, fine-tune, run, merge, print_stats, param_count, tokenize, self_test, qa_test, json_test, test_all, print_results, sweep-lr, sweep-corpus, lr-range-test, runpod\ntrain flags: --new-epoch [N] (continue training saved model for N more epochs, default 1)\ntrain_fineweb [max_mb] — download FineWeb shard + train (default 128MB, cached in fineweb_cache/)\nfine-tune env vars: FINETUNE_PATH (default fine_tune/finetune_train.json), LR (default 1e-4), EPOCHS (default 3)\nlr-range-test env vars: LR_LO (default 1e-5), LR_HI (default 5e-2), MAX_BATCHES (default 500)");
    Ok(())
}
