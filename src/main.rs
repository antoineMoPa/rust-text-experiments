use std::fs;
use std::io::prelude::*;

use attention_predictor::{create_model, get_pretrained_dict};
use candle_core::Var;

use crate::{
    attention_predictor::{get_device, Model, FILE_PATH, LR},
    model_tests::{per_epoch_scores, print_results, qa_test, self_test, test_all},
    token_utils::{tokenize, STOP_TOKEN},
};

mod attention_block;
mod attention_predictor;
mod grad_accum;
mod layer_norm;
mod model_tests;
mod models;
mod token_utils;

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
        let (dict, _) = get_pretrained_dict(FILE_PATH)?;
        let model = create_model(&dict, &device)?;
        println!("\nParameter count (vocab_size={}):", dict.len());
        model.count_params();
        return Ok(());
    }

    if command == "train" {
        println!("Training new model");

        let device = get_device()?;
        let (dict, _tokens) = get_pretrained_dict(FILE_PATH)?;
        let mut model = create_model(&dict, &device)?;

        let level_file_path = FILE_PATH;
        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;
        let tokens = tokenize(content.as_str());

        println!("Training on {} tokens", tokens.len());

        model.simple_train(tokens, &device, LR)?;
        model.save_to_path("data/model");

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
        println!("Loading model");
        let model = Model::load_from_path("data/model", &device)?;

        let args = args[1..].to_vec();

        let mut input = args.join(" ") + " ";
        println!("Completing: '{:?}'", input);

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

    if command == "self_test" {
        self_test()?;
        return Ok(());
    }

    if command == "qa_test" {
        qa_test()?;
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
        let (dict, _) = get_pretrained_dict(FILE_PATH)?;

        let mut file = fs::File::open(FILE_PATH)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let tokens = tokenize(&content);

        // 12 log-spaced LRs from 3e-4 to 1e-2
        let n = 12usize;
        let lr_lo = 3e-4f64;
        let lr_hi = 1e-2f64;
        let lrs: Vec<f64> = (0..n)
            .map(|i| (lr_lo.ln() + i as f64 / (n - 1) as f64 * (lr_hi.ln() - lr_lo.ln())).exp())
            .collect();

        for lr in lrs {
            println!("=== sweep-lr: LR = {:.2e} ===", lr);
            let mut model = create_model(&dict, &device)?;
            model.simple_train(tokens.clone(), &device, lr)?;

            match per_epoch_scores(&model, &device) {
                Ok((l2, l3, qa)) => {
                    let entry = serde_json::json!({
                        "LR": lr,
                        "Self_Test_Score_L2": l2,
                        "Self_Test_Score_L3": l3,
                        "QA_Test_Score": qa,
                    });
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("lr_sweep.log")
                    {
                        use std::io::Write as W;
                        let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap());
                    }
                    println!(
                        "sweep result: LR={:.2e} L2={:.3} L3={:.3} QA={:.3}",
                        lr, l2, l3, qa
                    );
                }
                Err(e) => eprintln!("sweep score failed for LR={:.2e}: {}", lr, e),
            }
        }

        return Ok(());
    }

    if command == "sweep-corpus" {
        let (dict, _) = get_pretrained_dict(FILE_PATH)?;

        let mut file = fs::File::open(FILE_PATH)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let tokens = tokenize(&content);

        // 10 evenly-spaced rates: 100%, 90%, ..., 10%
        let rates: Vec<f64> = (1..=10).rev().map(|i| i as f64 * 0.1).collect();

        for rate in rates {
            let n = ((tokens.len() as f64) * rate).round() as usize;
            let subset = tokens[..n].to_vec();
            println!("=== sweep-corpus: rate={:.0}% tokens={} ===", rate * 100.0, n);
            let mut model = create_model(&dict, &device)?;
            model.simple_train(subset, &device, LR)?;

            match per_epoch_scores(&model, &device) {
                Ok((l2, l3, qa)) => {
                    let entry = serde_json::json!({
                        "Corpus_Rate": rate,
                        "Num_Tokens": n,
                        "Self_Test_Score_L2": l2,
                        "Self_Test_Score_L3": l3,
                        "QA_Test_Score": qa,
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
                        "sweep result: rate={:.0}% tokens={} L2={:.3} L3={:.3} QA={:.3}",
                        rate * 100.0, n, l2, l3, qa
                    );
                }
                Err(e) => eprintln!("sweep score failed for rate={:.0}%: {}", rate * 100.0, e),
            }
        }

        return Ok(());
    }

    println!("Usage: rust-text-experiments <command>\nCommands: train, run, merge, print_stats, param_count, self_test, qa_test, test_all, print_results, sweep-lr, sweep-corpus");
    Ok(())
}
