use std::fs;

use crate::{token_utils::{tokenize, tokens_to_dict}, candle_predictor::{create_and_train_predictor_model, get_device}};

mod embedding_utils;
mod token_utils;
mod candle_autoencoder;
mod candle_predictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read args
    let file_path = "data/corpus/wiki-horse.txt";
    let content = fs::read_to_string(file_path)?;
    let tokens: Vec<String> = tokenize(&content)[0..400].to_vec();
    let dict = tokens_to_dict(tokens.clone());

    let args: Vec<String> = std::env::args().collect();
    let args = args[1..].to_vec();

    let device = get_device()?;
    let model = create_and_train_predictor_model(dict, tokens, &device)?;

    let input = args.join(" ") + " ";
    println!("Predicting next token for: '{:?}'", input);

    let pred = model.predict_next_token(input.as_str(), &device)?;

    println!("Predicted next token: {:?}", pred);

    Ok(())
}
