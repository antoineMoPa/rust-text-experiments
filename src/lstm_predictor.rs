use candle_core::{Device, Tensor, DType, Module};
use candle_nn as nn;
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot, RNN, rnn::LSTMState};

use crate::token_utils::{Dict, GetTokenEmbedding, tokenize, EMBEDDING_SIZE};

pub struct Mlp {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub lstm: nn::LSTM,
    pub state: nn::rnn::LSTMState,
    pub var_map: VarMap,
    pub dict: Dict,
}

impl Mlp {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder) -> Result<Self, candle_core::Error> {

        let lstm_size = 32;
        //let lstm_size = dict.len();


        let fc1 = nn::linear(EMBEDDING_SIZE as usize, EMBEDDING_SIZE,vb.pp("fc1"))?;
        let fc2 = nn::linear(lstm_size, dict.len(),vb.pp("fc2"))?;


        let lstm = nn::lstm(
            EMBEDDING_SIZE as usize,
            lstm_size,
            nn::LSTMConfig::default(),
            vb.pp("lstm"),
        )?;

        let state = lstm.zero_state(1)?;

        Ok(Self { fc1, fc2, state, lstm, dict, var_map })
    }

    fn reset(&mut self) -> Result<(), candle_core::Error> {
        self.state = self.lstm.zero_state(1)?;
        Ok(())
    }

    fn forward(&mut self, inputs: &Vec<Tensor>) -> Result<Tensor, candle_core::Error> {
        self.reset()?;

        for i in 0..inputs.len() {
            let input = (&inputs[i] * 0.5)?.unsqueeze(0)?;
            self.state = self.lstm.step(&input, &self.state)?;
        }

        let result = self.state.h().clone();
        let result = self.fc2.forward(&result)?;
        let result = result.squeeze(0)?;

        let result = result.tanh()?;
        //let result = nn::ops::softmax(&result, 0)?;

        return Ok(result);
    }

    fn run(&mut self, input_embedding: &Vec<Vec<f32>>, device: &Device) -> Result<String, candle_core::Error> {
        let inputs: Vec<Tensor> = input_embedding.iter().map(|input| Tensor::new(input.clone(), &device)).collect::<Result<Vec<Tensor>, candle_core::Error>>()?;

        let output_prob = self.forward(&inputs)?;
        let output_prob_max_index = output_prob.argmax(0)?;
        let n = output_prob_max_index.to_vec0::<u32>()?;
        let max_token = self.dict.iter().nth(n as usize).unwrap().0.clone();

        return Ok(max_token);
    }

    pub fn predict_next_token(&mut self, input: &str, device: &Device) -> Result<String, candle_core::Error> {
        let tokens_chain = tokenize(&input);
        let input = tokens_chain.to_vec();

        let input_embedding: Vec<Vec<f32> > = input.iter().map(|token| self.dict.get_token_embedding(token)).collect();

        self.run(&input_embedding, device)
    }
}

pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, train: bool, device: &Device) -> Result<Mlp, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = Mlp::new(dict, varmap, vb)?;

    // Optimizer settings
    let epochs = 20;
    let lr = 0.1;

    let params = ParamsAdamW {
        lr,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(model.var_map.all_vars(), params)?;

    if !train {
        return Ok(model);
    }

    let mut inputs: Vec<Vec<Tensor>> = Vec::new();
    let mut targets: Vec<Tensor> = Vec::new();

    for i  in 0..tokens_chain.len() {
        let chain = tokens_chain.to_vec();
        let context_window = 5;
        let start = ((i as i32) - context_window).max(0) as usize;
        let input = chain[start..i].to_vec();
        let output = chain[i].clone();


        // print input and output
        println!("input: {:?}, output: {:?}", input, output);

        let input: Vec<Tensor> = input.iter().map(|token| Tensor::new(model.dict.get_token_embedding(token), &device)).collect::<Result<Vec<Tensor>, candle_core::Error>>()?;

        let output = model.dict.get_word_index(output.as_str())?;

        let output = one_hot(Tensor::new(output, &device)?, model.dict.len(), 1.0 as f32, 0.0 as f32)?;

        inputs.push(input);
        targets.push(output);
    }

    // Training loop
    for epoch in 0..epochs {
        let mut outputs: Vec::<Tensor> = Vec::new();

        for (index, input) in inputs.iter().enumerate() {
            let output = model.forward(&input)?;
            outputs.push(output);

            //let loss = nn::loss::mse(&output, &targets[index])?;

            //optimizer.backward_step(&loss)?;

            //if epoch % 2 == 0 {
            //    println!("Epoch {}: Loss = {:?}", epoch, loss);
            //}
        }

        let outputs = Tensor::stack(&outputs, 0)?;
        let targets = Tensor::stack(&targets, 0)?;
        let loss = nn::loss::mse(&outputs, &targets)?;

        optimizer.backward_step(&loss)?;

        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss);
        }
    }

    Ok(model)
}

pub fn get_device() -> Result<Device, candle_core::Error> {
    let device = Device::Cpu;
    return Ok(device);

    // let device = Device::new_metal(0)?;
    // let metal_device = match &device {
    //     Device::Metal(m) => m,
    //     _ => panic!("Device is not Metal"),
    // };
    // return Ok(device);
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::token_utils::{tokenize, tokens_to_dict};

    use super::*;

    #[test]
    fn test_lstm_predictor_hello_world() -> Result<(), candle_core::Error> {
        let tokens = tokenize("hello world");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    fn test_lstm_predictor_lorem_1() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et dolor sit");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("ipsum ", &device)?, "et");

        Ok(())
    }

        #[test]
    fn test_lstm_predictor_lorem_2() -> Result<(), candle_core::Error> {
        let tokens = tokenize("lorem ipsum et dolor sit amet");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        assert_eq!(model.predict_next_token("lorem", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ", &device)?, "ipsum");
        assert_eq!(model.predict_next_token("ipsum ", &device)?, "et");
        assert_eq!(model.predict_next_token("dolor ", &device)?, "sit");

        assert_eq!(model.predict_next_token("lorem ipsum", &device)?, " ");
        assert_eq!(model.predict_next_token("lorem ipsum ", &device)?, "et");
        assert_eq!(model.predict_next_token("lorem ipsum et ", &device)?, "dolor");
        assert_eq!(model.predict_next_token("ipsum et ", &device)?, "dolor");

        Ok(())
    }

    #[test]
    fn test_horse_10() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..10].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_20() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..20].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("(Equus ", &device)?, "ferus");

        Ok(())
    }

    #[test]
    fn test_horse_40() -> Result<(), candle_core::Error> {
        // Define the file path
        let file_path = "data/corpus/wiki-horse.txt";
        let content = fs::read_to_string(file_path)?;
        let tokens: Vec<String> = tokenize(&content)[0..40].to_vec();

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;

        let substring = tokens[35..38].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);

        let substring = tokens[35..39].to_vec().join("");
        assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[39]);

        Ok(())
    }

    // #[test]
    // fn test_horse_60() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..60].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //     let substring = tokens[51..56].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[56]);
    //
    //     let substring = tokens[51..57].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[57]);
    //
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_horse_100() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..100].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //     let substring = tokens[63..70].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[70]);
    //
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_horse_200() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..200].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //     let substring = tokens[102..113].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[113]);
    //
    //     let substring = tokens[102..114].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);
    //
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_horse_400() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..400].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //
    //     let substring = tokens[102..114].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[114]);
    //
    //     let substring = tokens[162..182].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[182]);
    //
    //     let substring = tokens[190..211].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[211]);
    //
    //     let substring = tokens[330..341].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);
    //
    //     let substring = tokens[330..342].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[342]);
    //
    //     Ok(())
    // }
    //
    //     #[test]
    // fn test_horse_1000() -> Result<(), candle_core::Error> {
    //     // Define the file path
    //     let file_path = "data/corpus/wiki-horse.txt";
    //     let content = fs::read_to_string(file_path)?;
    //     let tokens: Vec<String> = tokenize(&content)[0..1000].to_vec();
    //
    //     let dict = tokens_to_dict(tokens.clone());
    //
    //     let device = get_device()?;
    //
    //     let mut model = create_and_train_predictor_model(dict, tokens.clone(), true, &device)?;
    //
    //     let substring = tokens[35..38].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[38]);
    //
    //
    //     let substring = tokens[63..69].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[69]);
    //
    //     let substring = tokens[330..341].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[341]);
    //
    //     let substring = tokens[810..831].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[831]);
    //
    //     let substring = tokens[810..832].to_vec().join("");
    //     assert_eq!(model.predict_next_token(substring.as_str(), &device)?, tokens[832]);
    //
    //     Ok(())
    // }
}
