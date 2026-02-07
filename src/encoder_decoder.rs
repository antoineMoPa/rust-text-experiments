use candle_core::{Device, Tensor, D};
use candle_nn::VarMap;
use candle_nn::{self as nn, Module};
use nn::encoding::one_hot;
use nn::{AdamW, Optimizer, VarBuilder};
use std::collections::BTreeMap;
use std::fs;

use crate::attention_predictor::{get_pretrained_dict, FILE_PATH};
use crate::token_utils::{tokens_to_dict, Dict, GetTokenEmbedding};

pub const EMBEDDING_SIZE: usize = 252;

pub struct EncoderDecoder {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc3: nn::Linear,
    pub var_map: VarMap,
    pub dict: Dict,
    pub device: Device,
    pub token_index: BTreeMap<String, u32>,
}

impl EncoderDecoder {
    pub fn new(
        dict: Dict,
        var_map: VarMap,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self, candle_core::Error> {
        let hidden_size = EMBEDDING_SIZE;

        let fc1 = nn::linear_b(dict.len(), hidden_size, false, vb.pp("fc1"))?;
        let fc2 = nn::linear_b(hidden_size, hidden_size, false, vb.pp("fc2"))?;
        let fc3 = nn::linear_b(hidden_size, dict.len(), false, vb.pp("fc3"))?;
        let device = device.clone();

        let token_index = dict.build_index();

        Ok(Self {
            fc1,
            fc2,
            fc3,
            dict,
            var_map,
            token_index,
            device,
        })
    }

    pub fn get_token_embedding(&self, token: &str) -> Result<Tensor, candle_core::Error> {
        let tensor = self.token_to_tensor(token)?;

        let result = self.fc1.forward(&tensor)?;
        let result = result.tanh()?;
        let result = self.fc2.forward(&result)?;
        let result = result.tanh()?;

        return Ok(result);
    }

    pub fn unembed(&self, tensor: &Tensor) -> Result<Tensor, candle_core::Error> {
        let result = self.fc3.forward(&tensor)?;
        let result = result.tanh()?;
        return Ok(result);
    }

    pub fn get_token_embedding_vec(&self, token: &str) -> Result<Vec<f32>, candle_core::Error> {
        let result = self.get_token_embedding(token)?;
        let result = result.squeeze(0)?;
        let vec = result.to_vec1::<f32>()?;

        return Ok(vec);
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let result = self.fc1.forward(&input)?;
        let result = nn::ops::dropout(&result, 0.3)?;
        let result = result.tanh()?;
        let result = self.fc2.forward(&result)?;
        let result = result.tanh()?;
        let result = self.fc3.forward(&result)?;
        let result = result.tanh()?;

        return Ok(result);
    }

    fn run(&self, input: String) -> Result<String, candle_core::Error> {
        let input: &Tensor = &self.token_to_tensor(input.as_str())?;
        let output_prob = self.forward(input)?;
        let output_prob_max_index = output_prob.argmax(1)?;

        let n = output_prob_max_index.to_vec1::<u32>()?[0];

        let max_token = self.dict.iter().nth(n as usize).unwrap();

        return Ok(max_token.0.clone());
    }

    pub fn token_to_tensor(&self, input: &str) -> Result<Tensor, candle_core::Error> {
        let token_index = *self.token_index.get(input).unwrap();
        let arr = vec![token_index as u32];
        let input = one_hot(
            Tensor::new(arr, &self.device)?,
            self.dict.len(),
            1.0 as f32,
            0.0 as f32,
        )?;

        return Ok(input);
    }

    pub fn train(&mut self) -> Result<(), candle_core::Error> {
        let epochs = 40;
        let lr = 0.003;
        let batch_size = 100;
        let mut optimizer: AdamW = AdamW::new_lr(self.var_map.all_vars(), lr)?;

        for epoch in 0..epochs {
            let last_batch = self.dict.len() / batch_size + 1;
            for i in 0..last_batch {
                let mut inputs = Vec::new();

                for (token, _token_index) in self.dict.iter().skip(i * batch_size).take(batch_size)
                {
                    let input = self.token_to_tensor(token)?;

                    inputs.push(input);
                }

                if inputs.len() == 0 {
                    continue;
                }

                let inputs = Tensor::stack(&inputs, 0)?;
                let targets = inputs.clone();

                let predictions = self.forward(&inputs)?;

                // Compute loss
                let loss = nn::loss::mse(&predictions, &targets)?;

                // Backpropagation
                optimizer.backward_step(&loss)?;

                if epoch % 10 == 0 {
                    println!(
                        "Epoch {:6}: Loss = {:.6} {}/{}",
                        epoch,
                        loss.to_vec0::<f32>()?,
                        i,
                        last_batch
                    );
                    self.evaluate()?;
                }
            }
        }

        Ok(())
    }

    /// Idea:
    ///
    /// In a neural network where the input is 3 word, train the network to output the middle word, then gradually remove the first and last word.
    ///
    pub fn train_with_corpus(&mut self) -> Result<(), candle_core::Error> {
        let (_dict, tokens) = get_pretrained_dict(FILE_PATH)?;
        let lr = 0.003;
        let mut optimizer: AdamW = AdamW::new_lr(self.var_map.all_vars(), lr)?;
        let batch_size = 100;
        let last_batch = tokens.len() / batch_size + 1;
        let epochs = 10;
        for epoch in 0..epochs {
            for i in 0..(last_batch + 1) {
                let mut inputs = Vec::new();
                let start = i * batch_size + 1;
                let end = (start + batch_size).min(tokens.len()) - 1;

                // Gradually remove previous and next token from input
                let f = match epoch {
                    0 => -0.1,
                    1 => -0.2,
                    2 => -0.3,
                    3 => -0.4,
                    4 => -0.5,
                    5 => -0.6,
                    6 => -0.5,
                    7 => -0.4,
                    8 => -0.3,
                    9 => -0.2,
                    10 => -0.1,
                    _ => 0.0,
                };

                for w in start..end {
                    let previous_token = tokens[w - 1].clone();
                    let token = tokens[w].clone();
                    let next_token = tokens[w + 1].clone();

                    let previous = self.token_to_tensor(previous_token.as_str())? * f;
                    let input = self.token_to_tensor(token.as_str())?;
                    let next = self.token_to_tensor(next_token.as_str())? * f;

                    let input = ((input + previous?)? + next?)?;

                    inputs.push(input);
                }

                if inputs.len() == 0 {
                    continue;
                }

                let inputs = Tensor::stack(&inputs, 0)?;
                let targets = inputs.clone();

                let predictions = self.forward(&inputs)?;

                // Compute loss
                let loss = nn::loss::mse(&predictions, &targets)?;

                // Backpropagation
                optimizer.backward_step(&loss)?;

                if i % 10 == 0 {
                    print!(
                        "Epoch {:3}/{:3} Batch {:4}/{:4}: Loss = {:.6} f = {:.2} -    ",
                        epoch,
                        epochs,
                        i,
                        last_batch,
                        loss.to_vec0::<f32>()?,
                        f
                    );
                    self.evaluate()?;
                }
            }
        }

        Ok(())
    }

    pub fn train_strategy(&mut self) -> Result<(), candle_core::Error> {
        self.train()?;
        self.train_with_corpus()?;
        self.train()?;
        self.train_with_corpus()?;
        self.train()?;

        if self.evaluate()?.1 > 0 {
            panic!("EncodeDecoder failed to encode dictionnary");
        }

        Ok(())
    }

    pub fn evaluate(&self) -> Result<(f32, i32), candle_core::Error> {
        let mut successes = 0;
        let mut failures = 0;

        for token in self.dict.keys() {
            let result = self.run(token.clone())?;

            if &result == token {
                successes += 1;
            } else {
                failures += 1;
            }
        }
        println!("Successes: {}, Failures: {}", successes, failures);

        let success_rate = successes as f32 / (successes + failures) as f32;

        return Ok((success_rate, failures));
    }

    pub fn save_to_path(&self, path: &str) {
        let var_map_path = format!("{}.safetensors", path);
        self.var_map.save(var_map_path.as_str()).unwrap();

        let dict_words = self
            .dict
            .iter()
            .map(|(word, _)| word.clone())
            .collect::<Vec<String>>();

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

        let mut model = EncoderDecoder::new(dict, varmap, vb, &device)?;
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

    pub fn build_token_embedding_map(
        &self,
    ) -> Result<BTreeMap<String, Vec<f32>>, candle_core::Error> {
        let mut token_embedding_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let tokens_chain: Vec<String> = self.dict.keys().cloned().collect();
        let unique_tokens: Vec<String> = tokens_chain
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();
        for token in unique_tokens {
            let token_embedding = self.get_token_embedding_vec(token.as_str())?;
            token_embedding_map.insert(token, token_embedding);
        }

        return Ok(token_embedding_map);
    }

    pub fn build_token_embedding_tensor_map(
        &self,
    ) -> Result<BTreeMap<String, Tensor>, candle_core::Error> {
        let mut token_embedding_map: BTreeMap<String, Tensor> = BTreeMap::new();
        let tokens_chain: Vec<String> = self.dict.keys().cloned().collect();
        let unique_tokens: Vec<String> = tokens_chain
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<String>>()
            .into_iter()
            .collect();
        for token in unique_tokens {
            let token_embedding = self.get_token_embedding(token.as_str())?;
            token_embedding_map.insert(token, token_embedding);
        }

        return Ok(token_embedding_map);
    }

    pub fn print_stats(&self) -> Result<(), candle_core::Error> {
        println!("Model stats:");
        println!("Dict size: {}", self.dict.len());

        // print min, max, mean, std of all tensors
        for var in self.var_map.all_vars().iter() {
            let min = var.min_all()?.to_vec0::<f32>()?;
            let max = var.max_all()?.to_vec0::<f32>()?;
            let mean = var.mean_all()?.to_vec0::<f32>()?;
            let variance = var.flatten_all()?.var(D::Minus1)?.to_vec0::<f32>()?;
            let abs_min = var.abs()?.min_all()?.to_vec0::<f32>()?;
            let abs_max = var.abs()?.max_all()?.to_vec0::<f32>()?;

            println!(
                "min: {:.3}, max: {:.3}, mean: {:.3}, std: {:.3} abs_min: {:.4} abs_max: {:.4}",
                min, max, mean, variance, abs_min, abs_max
            );
        }

        Ok(())
    }

    pub fn print_dict_embeddings(&self) -> Result<(), candle_core::Error> {
        for word in self.dict.keys() {
            let embedding: Tensor = self.get_token_embedding(word)?.squeeze(0)?;
            let embedding = embedding.to_vec1::<f32>()?;
            println!("{}: {:?}", word, embedding);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::{attention_predictor::FILE_PATH, token_utils::tokenize};

    use super::*;

    #[test]
    fn test_encoder_decoder() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("Hello, world!");
        let dict = tokens_to_dict(vocabulary);
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        encoder_decoder.train()?;

        encoder_decoder.save_to_path("data/encdec");
        let encoder_decoder = EncoderDecoder::load_from_path("data/encdec", &device)?;

        assert_eq!(encoder_decoder.run("Hello".to_string())?, "Hello");
        assert_eq!(encoder_decoder.run("world".to_string())?, "world");
        assert_eq!(encoder_decoder.run(" ".to_string())?, " ");

        Ok(())
    }

    #[test]
    fn test_encoder_decoder_lorem() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("Lorem ipsum et dolor sit amet.");
        let dict = tokens_to_dict(vocabulary);
        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        encoder_decoder.train()?;

        assert_eq!(encoder_decoder.run("Lorem".to_string())?, "Lorem");
        assert_eq!(encoder_decoder.run("ipsum".to_string())?, "ipsum");
        assert_eq!(encoder_decoder.run(" ".to_string())?, " ");

        Ok(())
    }

    #[test]
    fn test_encoder_decoder_corpus_training() -> Result<(), candle_core::Error> {
        let (dict, _tokens) = get_pretrained_dict(FILE_PATH)?;

        let device = EncoderDecoder::get_device()?;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        encoder_decoder.train_strategy()?;

        let success_rate = encoder_decoder.evaluate()?;

        assert!(success_rate.0 > 0.9);

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_encoder_decoder_level0() -> Result<(), candle_core::Error> {
        let level_file_path = "common-corpus/level_2/corpus.corpus";
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
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        println!("Training");
        encoder_decoder.train()?;

        let success_rate = encoder_decoder.evaluate()?;

        assert!(success_rate.0 > 0.9);

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_encoder_decoder_whole_corpus() -> Result<(), candle_core::Error> {
        let level_file_path = FILE_PATH;
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
        let mut encoder_decoder = EncoderDecoder::new(dict, vm, vb, &device)?;

        println!("Training");
        encoder_decoder.train()?;
        encoder_decoder.save_to_path("data/encdec");

        let success_rate = encoder_decoder.evaluate()?;

        assert!(success_rate.0 > 0.9);

        Ok(())
    }
}
