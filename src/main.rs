use std::fs;
use std::io::prelude::*;

use attention_predictor::{create_model, get_pretrained_dict};
use candle_core::Var;

use crate::{
    attention_predictor::{get_device, Model, FILE_PATH},
    model_tests::{qa_test, self_test},
    token_utils::{tokenize, STOP_TOKEN},
};

mod attention_block;
mod attention_predictor;
mod grad_accum;
mod model_tests;
mod models;
mod token_utils;

fn read_n_chars(file_path: &str, n: u64) -> Result<String, std::io::Error> {
    let file = fs::File::open(file_path)?;
    let mut content = String::new();
    let mut handle = file.take(n);

    handle.read_to_string(&mut content)?;

    Ok(content)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let command = args.first().map(|s| s.as_str()).unwrap_or("");

    let device = get_device()?;

    // idea: pre train with a 300 words
    // see if it responds well in a number of ways:
    // can learn hello world
    // can output a sequence of words + spaces
    // then train with larger dataset if the model is a good one.

    if command == "print_stats" {
        println!("Loading test model");
        let model = Model::load_from_path("data/model", &device)?;
        model.print_stats()?;
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

        model.simple_train(tokens, &device)?;
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

        let mut buf = String::new();
        loop {
            let pred = model.predict_next_token(input.as_str(), &device)?;

            if pred == STOP_TOKEN {
                println!("{} - ", buf);
                buf.clear();
                break;
            }

            input = input + pred.as_str();

            buf.push_str(pred.as_str());

            if buf.len() > 40 {
                println!("{} - ", buf);
                buf.clear()
            }
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

    println!("Usage: rust-text-experiments <command>\nCommands: train, run, merge, print_stats, self_test, qa_test");
    Ok(())
}
