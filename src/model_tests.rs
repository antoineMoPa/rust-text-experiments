use std::fs;
use std::io::prelude::*;

use crate::{
    attention_predictor::{get_device, Model},
    token_utils::STOP_TOKEN,
};

const RESULT_COLS: &[&str] = &[
    "Model_ID",
    "Corpus_Level",
    "Dict_Size",
    "Embedding_Size",
    "Context_Window",
    "Epochs",
    "Hidden_Size",
    "Num_blocks",
    "Num_att_heads",
    "LR",
    "Batch_Size",
    "State_of_the_code",
    "Time_to_train",
    "Self_Test_Score_L2",
    "Self_Test_Score_L3",
    "QA_Test_Score",
    "Date",
];

fn read_ndjson(path: &str) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub fn print_results() -> Result<(), Box<dyn std::error::Error>> {
    let train = read_ndjson("training_log.json");
    let tests = read_ndjson("test_results.json");

    println!("{}", RESULT_COLS.join(","));

    let empty = serde_json::Value::Object(Default::default());
    for t in &train {
        // Match test result by Model_ID, fall back to empty if not found
        let r = t
            .get("Model_ID")
            .and_then(|id| tests.iter().rev().find(|r| r.get("Model_ID") == Some(id)))
            .unwrap_or(&empty);
        let row: Vec<String> = RESULT_COLS
            .iter()
            .map(|col| {
                t.get(*col)
                    .or_else(|| r.get(*col))
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default()
            })
            .collect();
        println!("{}", row.join(","));
    }

    Ok(())
}

pub fn self_test() -> Result<(), Box<dyn std::error::Error>> {
    let (l2, l3) = self_test_scores()?;
    println!("Self test scores: L2={}, L3={}", l2, l3);
    Ok(())
}

pub fn qa_test() -> Result<(), Box<dyn std::error::Error>> {
    let score = qa_test_score()?;
    println!("QA test score: {}", score);
    Ok(())
}

pub fn test_all() -> Result<(), Box<dyn std::error::Error>> {
    let (score_l2, score_l3) = self_test_scores()?;
    let score_qa = qa_test_score()?;

    let date = std::process::Command::new("date")
        .arg("+%d/%m/%Y")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let model_id = std::fs::read_to_string("data/model.id")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let entry = serde_json::json!({
        "Model_ID": model_id,
        "Self_Test_Score_L2": score_l2,
        "Self_Test_Score_L3": score_l3,
        "QA_Test_Score": score_qa,
        "Date": date,
    });

    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("test_results.json")
    {
        writeln!(file, "{}", serde_json::to_string(&entry).unwrap())?;
    }

    println!("Self_Test_Score_L2\tSelf_Test_Score_L3\tQA_Test_Score\tDate");
    println!("{}\t{}\t{}\t{}", score_l2, score_l3, score_qa, date);

    Ok(())
}

fn first_n_words_contain(output: &str, expected: &str, n: usize) -> bool {
    let prefix = output
        .split_whitespace()
        .take(n)
        .collect::<Vec<_>>()
        .join(" ");
    prefix.contains(expected)
}

fn self_test_scores() -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let device = get_device()?;
    println!("Loading test model");
    let model = Model::load_from_path("data/model", &device)?;

    let level_file_paths = vec![
        "common-corpus/level_2/corpus.txt",
        "common-corpus/level_3/corpus.txt",
    ];

    let mut scores: Vec<f32> = Vec::new();

    for level_file_path in level_file_paths.iter() {
        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;

        let mut match_count = 0;
        let mut total = 0;

        for line in content.split("\n").take(20) {
            let words: Vec<&str> = line.split(" ").take(6).collect();
            let expected_completion = line.split(" ").skip(6);
            let original_input = words.join(" ");
            let mut input = original_input.clone() + " ";

            let mut buf = String::new();
            loop {
                let pred = model.predict_next_token_greedy(input.as_str(), &device)?;
                input = input + pred.as_str();

                if buf.len() > 50 || pred == "." {
                    buf.push_str(pred.as_str());
                    break;
                }

                if pred == STOP_TOKEN {
                    break;
                }

                buf.push_str(pred.as_str());
            }

            let expected_completion: Vec<&str> = expected_completion.collect();
            let expected_completion = expected_completion.join(" ").replace("<stop>", "");

            if first_n_words_contain(&buf, &expected_completion, 3) {
                match_count += 1;
                println!(
                    "'{}' |> '{}' ~ '{}' - match",
                    original_input, buf, expected_completion
                );
            } else {
                println!(
                    "'{}' |> '{}' ~ '{}' - no match",
                    original_input, buf, expected_completion
                );
            }

            total += 1;
        }

        let success_rate = match_count as f32 / total as f32;
        println!(
            "corpus {} - matches - {}, total - {}, success rate - {}",
            level_file_path, match_count, total, success_rate
        );
        scores.push(success_rate);
    }

    Ok((scores[0], scores[1]))
}

fn qa_test_score() -> Result<f32, Box<dyn std::error::Error>> {
    let device = get_device()?;
    println!("Loading test model");
    let model = Model::load_from_path("data/model", &device)?;

    let file_path = "common-corpus/level_3/qa.txt";
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
            let pred = model.predict_next_token_greedy(&input, &device)?;
            input = input + pred.as_str();

            if buf.len() > 50 || pred == "." {
                buf.push_str(pred.as_str());
                break;
            }

            if pred == STOP_TOKEN {
                break;
            }

            buf.push_str(pred.as_str());
        }

        let buf = buf.trim().to_lowercase().replace(".", "");
        let answer = answer.trim().to_lowercase().replace(".", "");

        if first_n_words_contain(&buf, &answer, 3) {
            match_count += 1;
            println!("'{}' |> '{}' ~ '{}' - match", question, buf, answer);
        } else {
            println!("'{}' |> '{}' ~ '{}' - no match", question, buf, answer);
        }

        total += 1;
    }

    let success_rate = match_count as f32 / total as f32;
    println!(
        "question file: {} - matches - {}, total - {}, success rate - {}",
        file_path, match_count, total, success_rate
    );

    Ok(success_rate)
}
