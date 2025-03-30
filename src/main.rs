use std::fs;
use std::io::prelude::*;

use attention_predictor::{create_model, get_pretrained_dict};
use candle_nn::{VarMap, VarBuilder};

use crate::{
    token_utils::{tokenize, STOP_TOKEN},
    attention_predictor::{get_device, Model, FILE_PATH}, encoder_decoder::EncoderDecoder
};

mod token_utils;
mod attention_block;
mod simple_predictor;
mod lstm_predictor;
mod encoder_decoder;
mod attention_predictor;
mod models;

fn read_n_chars(file_path: &str, n: u64) -> Result<String, std::io::Error> {
    let file = fs::File::open(file_path)?;
    let mut content = String::new();
    let mut handle = file.take(n);

    handle.read_to_string(&mut content)?;

    Ok(content)
}

fn self_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = get_device()?;
    println!("Loading test model");
    let model = Model::load_from_path("data/model", &device)?;

    let level_file_paths = vec![
        "data/corpus/level_2/corpus.txt",
        "data/corpus/level_3/corpus.txt",
    ];

    for level_file_path in level_file_paths.iter() {
        let level_file_path = level_file_path;

        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;

        let mut match_count = 0;
        let mut total = 0;

        for line in content.split("\n") {
            let words: Vec<&str> = line.split(" ").take(3).collect();
            let expected_completion = line.split(" ").skip(3);
            let original_input = words.join(" ");
            let mut input = original_input.clone() + " ";

            let mut buf = String::new();
            loop {
                let pred = model.predict_next_token(input.as_str(), &device)?;
                input = input + pred.as_str();

                buf.push_str(pred.as_str());

                if buf.len() > 50 || pred == "." {
                    break;
                }
            }

            let expected_completion: Vec<&str> = expected_completion.collect();
            let expected_completion = expected_completion.join(" ").replace("<stop>", "");

            if buf == expected_completion {
                match_count += 1;
                println!("'{}' |> '{}' ~ '{}' - match", original_input, buf, expected_completion);
            }
            else {
                println!("'{}' |> '{}' ~ '{}' - no match", original_input, buf, expected_completion);
            }

            total += 1;
        }

        let success_rate = match_count as f32 / total as f32;

        println!("corpus {} - matches - {}, total - {}, success rate - {}", level_file_path, match_count, total, success_rate);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let args: Vec<String> = std::env::args().collect();
    let args = args[1..].to_vec();

    let device = get_device()?;

    // idea: pre train with a 300 words
    // see if it responds well in a number of ways:
    // can learn hello world
    // can output a sequence of words + spaces
    // then train with larger dataset if the model is a good one.

    if args[0] == "pretrain_encoder_decoder" {
        let level_file_path = FILE_PATH;
        let (dict, _tokens) = get_pretrained_dict(level_file_path)?;
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        println!("Training encoder decoder");
        encoder_decoder.train_strategy()?;
        encoder_decoder.save_to_path("data/encdec");

        encoder_decoder.evaluate()?;

        return Ok(());
    }

    if args[0] == "print_stats" {
        println!("Loading test model");
        let model = Model::load_from_path("data/model", &device)?;
        model.print_stats()?;
        return Ok(());
    }
    if args[0] == "print_dict_embeddings" {
        println!("Loading test model");
        let model = EncoderDecoder::load_from_path("data/encdec", &device)?;
        model.print_dict_embeddings()?;
        return Ok(());
    }

    if args[0] == "print_stats_encoder_decoder" {
        println!("Loading encoder decoder model");
        let model = EncoderDecoder::load_from_path("data/encdec", &device)?;
        model.print_stats()?;
        return Ok(());
    }

    if args[0] == "pretrain" {
        println!("Pretraining test model");

        let device = get_device()?;
        let mut model = create_model(&device)?;

        // in case we want to continue training:
        // model.load_inplace_from_path("data/model")?;

        // train on data/corpus/level_/corpus.txt
        let level_file_path = FILE_PATH;
        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;
        let tokens = tokenize(content.as_str());

        println!("Training level 0 on {} tokens", tokens.len());

        model.simple_train(tokens, &device)?;
        model.save_to_path("data/model");

        return Ok(());
    }

    if args[0] == "run" {
        println!("Loading test model");
        let model = Model::load_from_path("data/model", &device)?;

        let args = args[1..].to_vec();

        let mut input = args.join(" ") + " ";
        println!("Predicting next token for: '{:?}'", input);

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
    }

    if args[0] == "self_test" {
        self_test()?;
        return Ok(());
    }


    println!("Please provide a valid command: 'train' or 'run'");
    Ok(())
}
