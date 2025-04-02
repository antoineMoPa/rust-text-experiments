use std::fs;
use std::io::prelude::*;

use crate::attention_predictor::{get_device, Model};

pub fn self_test() -> Result<(), Box<dyn std::error::Error>> {
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

// Questions / Answers test
pub fn qa_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = get_device()?;
    println!("Loading test model");
    let model = Model::load_from_path("data/model", &device)?;

    let file_path = "data/corpus/level_3/qa.txt";
    let mut file = fs::File::open(file_path)?;
    let mut content: String = String::new();
    file.read_to_string(&mut content)?;

    let mut match_count = 0;
    let mut total = 0;

    for line in content.split("\n") {
        let question: Vec<&str> = line.split("A:").take(1).collect();
        let question = question.join("");
        let answer: Vec<&str> = line.split("A:").skip(1).take(1).collect();
        let answer = answer.join("");

        let mut buf = String::new();
        let mut input = question.clone() + "A: ";

        loop {
            let pred = model.predict_next_token(&input, &device)?;
            input = input + pred.as_str();

            buf.push_str(pred.as_str());

            if buf.len() > 50 || pred == "." {
                break;
            }
        }

        let buf = buf.trim().to_lowercase();
        let answer = answer.trim().to_lowercase();

        if buf == answer {
            match_count += 1;
            println!("'{}' |> '{}' ~ '{}' - match", question, buf, answer);
        }
        else {
            println!("'{}' |> '{}' ~ '{}' - no match", question, buf, answer);
        }

        total += 1;
    }

    let success_rate = match_count as f32 / total as f32;

    println!("question file: {} - matches - {}, total - {}, success rate - {}", file_path, match_count, total, success_rate);

    Ok(())
}
