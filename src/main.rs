use std::fs;
use std::io::prelude::*;

use crate::{
    token_utils::{tokenize, tokens_to_dict},
    attention_predictor::{create_and_train_predictor_model, get_device}
};

mod token_utils;
mod simple_predictor;
mod lstm_predictor;
mod attention_predictor;

fn read_n_chars(file_path: &str, n: u64) -> Result<String, std::io::Error> {
    let file = fs::File::open(file_path)?;
    let mut content = String::new();
    let mut handle = file.take(n);

    handle.read_to_string(&mut content)?;

    Ok(content)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read args
    let file_path = "data/corpus/blogtext.csv";
    // read 20k chars of the file
    let content = read_n_chars(file_path, 4000 * 5)?; // approx 4k words
    let tokens: Vec<String> = tokenize(&content).to_vec();
    let dict = tokens_to_dict(tokens.clone());

    let args: Vec<String> = std::env::args().collect();
    let args = args[1..].to_vec();

    let device = get_device()?;

    // idea: pre train with a 300 words
    // see if it responds well in a number of ways:
    // can learn hello world
    // can output a sequence of words + spaces
    // then train with larger dataset if the model is a good one.

    if args[0] == "train" {
        println!("Training model");

        let model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        model.var_map.save("data/model.safetensors")?;

        return Ok(());
    }

    if args[0] == "run" {
        println!("Loading model");

        let mut model = create_and_train_predictor_model(dict, tokens, false, &device)?;

        model.var_map.load("data/model.safetensors")?;

        let args = args[1..].to_vec();

        let mut input = args.join(" ") + " ";
        println!("Predicting next token for: '{:?}'", input);

        loop {
            let pred = model.predict_next_token(input.as_str(), &device)?;
            input = input + pred.as_str();
            print!("{}", pred);
        }
    }

    println!("Please provide a valid command: 'train' or 'run'");
    Ok(())
}
