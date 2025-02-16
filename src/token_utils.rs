use std::fs;
use std::io::prelude::*;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use candle_nn::{self as nn, Module};
use nn::{VarBuilder, AdamW, Optimizer};
use nn::encoding::one_hot;

pub type Dict = std::collections::BTreeMap<String, f32>;

pub const EMBEDDING_SIZE: usize = 80;

pub trait GetTokenEmbedding {
    fn get_token_embedding(&self, token: &str) -> Vec<f32>;
    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error>;
}

impl GetTokenEmbedding for Dict {
    fn get_token_embedding(&self, token: &str) -> Vec<f32> {
        let default = 0.0 as f32;
        let value = *self.get(token).unwrap_or(&default);
        let mut embedding: Vec<f32> = Vec::new();

        embedding.push(value);

        while embedding.len() < EMBEDDING_SIZE {
            for (_index, letter) in token.chars().enumerate() {
                if embedding.len() >= EMBEDDING_SIZE {
                    break;
                }

                let letter_value = letter as i32 as f32 / (EMBEDDING_SIZE as f32);

                embedding.push(letter_value.cos());
            }
        }

        assert_eq!(embedding.len(), EMBEDDING_SIZE);

        return embedding;
    }

    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error> {
        for (i, (current_token, _b)) in self.iter().enumerate() {
            if current_token == token {
                return Ok(i as u32);
            }
        }

        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Token not found"));
    }
}

pub fn tokenize(input: &str) -> Vec<String> {
    let split_symbols = [
        ' ',
        ',',
        '.',
        '!',
        '?',
        ';',
        ':',
        '\n',
        '\t',
        '(',
        ')',
        '{',
        '}',
        '[',
        ']',
        '<',
        '>',
        '=',
        '+',
        '-',
        '*',
        '/',
        '&',
        '|',
        '^',
        '%',
        '$',
    ];

    let mut tokens = Vec::new();
    let mut token = String::new();

    for c in input.chars() {
        if split_symbols.contains(&c) {
            if token.len() > 0 {
                tokens.push(token.clone());
                token.clear();
            }
            tokens.push(c.to_string());
        } else {
            token.push(c);
        }
    }

    if token.len() > 0 {
        tokens.push(token.clone());
    }

    return tokens;
}

pub fn create_vocabulary(tokens: Vec<String>) -> Vec<String> {
    let mut vocabulary = Vec::new();
    for token in tokens {
        if !vocabulary.contains(&token) {
            vocabulary.push(token);
        }
    }
    return vocabulary;
}


pub fn tokens_to_dict(vocabulary: Vec<String>) -> Dict {
    let mut vocabulary_dict = Dict::new();
    for (i, token) in vocabulary.iter().enumerate() {
        if vocabulary_dict.contains_key(token) {
            continue;
        }
        vocabulary_dict.insert(token.clone(), i as f32 / vocabulary.len() as f32);
    }
    return vocabulary_dict;
}

pub struct EncoderDecoder {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub var_map: VarMap,
    pub dict: Dict,
}

impl EncoderDecoder {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let hidden_size = 80;

        let fc1 = nn::linear_b(dict.len(), hidden_size, false, vb.pp("fc1"))?;
        let fc2 = nn::linear_b(hidden_size, hidden_size, false, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(hidden_size, dict.len(), false, vb.pp("fc3"))?;

        Ok(Self { fc1, fc2, fc3, dict, var_map })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let result = self.fc1.forward(&input)?;
        let result = result.tanh()?;
        let result = self.fc2.forward(&result)?;
        let result = result.tanh()?;
        let result = self.fc3.forward(&result)?;

        let result = result.tanh()?;

        return Ok(result);
    }

    fn run(&self, input: String, device: &Device) -> Result<String, candle_core::Error> {
        let input: &Tensor = &self.token_to_tensor(input.as_str(), &device)?;
        let output_prob = self.forward(input)?;
        let output_prob_max_index = output_prob.argmax(1)?;

        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn token_to_tensor(&self, input: &str, device: &Device) -> Result<Tensor, candle_core::Error> {
        let token_index = self.dict.get_word_index(input)?;
        let arr = vec![token_index as u32];
        let input = one_hot(Tensor::new(arr, &device)?, self.dict.len(), 0.95 as f32, 0.0 as f32)?;

        return Ok(input);
    }

    pub fn train(&mut self, device: &Device) -> Result<(), candle_core::Error> {
        // 1. More epoch when sample size is smaller
        let epochs = 10;

        let mut optimizer: AdamW = AdamW::new_lr(self.var_map.all_vars(), 0.003)?;

        for epoch in 0..epochs {
            let batch_size = 200;
            let last_batch = self.dict.len() / batch_size;
            for i in 0..last_batch {
                let mut inputs = Vec::new();

                for (token, _token_index) in self.dict.iter().skip(i*batch_size).take(batch_size) {
                    let input = self.token_to_tensor(token, device)?;

                    inputs.push(input);
                }

                let inputs = Tensor::stack(&inputs, 0)?;
                let targets = inputs.clone();

                let predictions = self.forward(&inputs)?;

                // Compute loss
                let loss = nn::loss::mse(&predictions, &targets)?;

                // Backpropagation
                optimizer.backward_step(&loss)?;

                if epoch % 1 == 0 {
                    println!("Epoch {:6}: Loss = {:.6} {}/{}", epoch, loss.to_vec0::<f32>()?, i, last_batch);

                    if i % 5 == 0 {
                        self.evaluate(device)?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn evaluate(&self, device: &Device) -> Result<f32, candle_core::Error> {
        let mut successes = 0;
        let mut failures = 0;

        for token in self.dict.keys() {
            let result = self.run(token.clone(), &device)?;

            if &result == token {
                successes += 1;
            } else {
                failures += 1;
            }
        }

        println!("Successes: {}, Failures: {}", successes, failures);

        let success_rate = successes as f32 / (successes + failures) as f32;

        return Ok(success_rate);
    }

    pub fn save_to_path(&self, path: &str) {
        let var_map_path = format!("{}.safetensors", path);
        self.var_map.save(var_map_path.as_str()).unwrap();

        let dict_words = self.dict.iter().map(|(word, _)| word.clone()).collect::<Vec<String>>();

        let dict_path = format!("{}.dict", path);
        let file = fs::File::create(dict_path).unwrap();
        serde_json::to_writer(file, &dict_words).unwrap();
    }

    pub fn load_from_path(path: &str, device: &Device) -> Result<Self, candle_core::Error> {
        let dict_path = format!("{}.dict", path);
        let file = fs::File::open(dict_path).unwrap();
        let dict_words: Vec<String> = serde_json::from_reader(file).unwrap();

        let dict = tokens_to_dict(dict_words);

        let varmap = VarMap::new();
        let var_map_path = format!("{}.safetensors", path);

        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let mut model = Self::new(dict, varmap, vb)?;
        model.var_map.load(var_map_path.as_str()).unwrap();

        return Ok(model);
    }

    pub fn get_device() -> Result<Device, candle_core::Error> {
        if cfg!(target_os = "macos") {
            let device = Device::new_metal(0)?;
            match &device {
                Device::Metal(m) => m,
                _ => panic!("Device is not Metal"),
            };
        return Ok(device);
        } else {
            return Device::new_cuda(0);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(tokenize("Hello, world!"), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_create_vocabulary() {
        let tokens = tokenize("Hello, world!");

        assert_eq!(create_vocabulary(tokens), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_vocabulary_to_dict() {
        let vocabulary = tokenize("Hello, world!");
        let vocabulary_dict = tokens_to_dict(vocabulary);

        assert_ne!(vocabulary_dict.get("Hello").unwrap(),
                   vocabulary_dict.get("world").unwrap());

        assert!(*vocabulary_dict.get("world").unwrap() > 0.0);
    }

    #[test]
    fn test_encoder_decoder() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("Hello, world!");
        let dict = tokens_to_dict(vocabulary);
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb)?;

        encoder_decoder.train(&device)?;

        encoder_decoder.save_to_path("data/encdec");
        let encoder_decoder = EncoderDecoder::load_from_path("data/encdec", &device)?;

        assert_eq!(encoder_decoder.run("Hello".to_string(), &device)?, "Hello");
        assert_eq!(encoder_decoder.run("world".to_string(), &device)?, "world");
        assert_eq!(encoder_decoder.run(" ".to_string(), &device)?, " ");

        Ok(())
    }

    #[test]
    fn test_encoder_decoder_lorem() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("Lorem ipsum et dolor sit amet.");
        let dict = tokens_to_dict(vocabulary);
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb)?;

        encoder_decoder.train(&device)?;

        assert_eq!(encoder_decoder.run("Lorem".to_string(), &device)?, "Lorem");
        assert_eq!(encoder_decoder.run("ipsum".to_string(), &device)?, "ipsum");
        assert_eq!(encoder_decoder.run(" ".to_string(), &device)?, " ");

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_encoder_decoder_level0() -> Result<(), candle_core::Error> {
        let level_file_path = "data/corpus/level_0/corpus.corpus";
        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;

        println!("Parsing content");
        let vocabulary = tokenize(content.as_str());
        println!("Building dictionary");
        let dict = tokens_to_dict(vocabulary.clone());
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb)?;

        println!("Training");
        encoder_decoder.train(&device)?;

        let success_rate = encoder_decoder.evaluate(&device)?;

        assert!(success_rate > 0.9);

        Ok(())
    }

    #[test]
    fn test_encoder_decoder_whole_corpus() -> Result<(), candle_core::Error> {
        let level_file_path = "data/corpus/corpus.txt";
        let mut file = fs::File::open(level_file_path)?;
        let mut content: String = String::new();
        file.read_to_string(&mut content)?;

        println!("Parsing content");
        let vocabulary = tokenize(content.as_str());
        println!("Building dictionary");
        let dict = tokens_to_dict(vocabulary.clone());
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb)?;

        println!("Training");
        encoder_decoder.train(&device)?;
        encoder_decoder.save_to_path("data/encdec");

        let success_rate = encoder_decoder.evaluate(&device)?;

        assert!(success_rate > 0.9);

        Ok(())
    }
}
