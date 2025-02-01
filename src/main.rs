use std::fs;

use crate::{
    token_utils::{tokenize, tokens_to_dict},
    lstm_predictor::{create_and_train_predictor_model, get_device}
};

mod token_utils;
//mod simple_predictor;
mod lstm_predictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read args
    let file_path = "data/corpus/wiki-horse.txt";
    let content = fs::read_to_string(file_path)?;
    let tokens: Vec<String> = tokenize(&content)[0..10].to_vec();
    let dict = tokens_to_dict(tokens.clone());

    let args: Vec<String> = std::env::args().collect();
    let args = args[1..].to_vec();

    let device = get_device()?;

    if args[0] == "train" {
        println!("Training model");

        let model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        model.var_map.save("data/horse.safetensors")?;

        return Ok(());
    }

    if args[0] == "run" {
        println!("Loading model");

        let mut model = create_and_train_predictor_model(dict, tokens, false, &device)?;

        model.var_map.load("data/horse.safetensors")?;

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
