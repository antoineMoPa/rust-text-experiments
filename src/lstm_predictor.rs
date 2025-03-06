#[cfg(test)]
use candle_core::{Device, Tensor, DType, Module};
#[cfg(test)]
use candle_nn as nn;
#[cfg(test)]
use nn::{VarMap, Optimizer, VarBuilder, ParamsAdamW, encoding::one_hot, RNN};

#[cfg(test)]
use crate::token_utils::{Dict, GetTokenEmbedding, tokenize, EMBEDDING_SIZE};

#[cfg(test)]
pub struct Mlp {
    pub fc2: nn::Linear,
    pub lstm: nn::LSTM,
    pub state: nn::rnn::LSTMState,
    pub var_map: VarMap,
    pub dict: Dict,
}

#[cfg(test)]
impl Mlp {
    pub fn new(dict: Dict, var_map: VarMap, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let lstm_size = 32;
        //let lstm_size = dict.len();

        let fc2 = nn::linear(lstm_size, dict.len(),vb.pp("fc2"))?;


        let lstm = nn::lstm(
            EMBEDDING_SIZE as usize,
            lstm_size,
            nn::LSTMConfig::default(),
            vb.pp("lstm"),
        )?;

        let state = lstm.zero_state(1)?;

        Ok(Self { fc2, state, lstm, dict, var_map })
    }

    fn reset(&mut self) -> Result<(), candle_core::Error> {
        self.state = self.lstm.zero_state(1)?;
        Ok(())
    }

    fn forward(&mut self, inputs: &Vec<Tensor>) -> Result<Tensor, candle_core::Error> {
        self.reset()?;

        for i in 0..inputs.len() {
            let input = (&inputs[i] * 0.4)?.unsqueeze(0)?;
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

        let input_embedding: Vec<Vec<f32> > = input.iter().map(|token| self.dict.get_token_cos_encoding(token)).collect();

        self.run(&input_embedding, device)
    }
}

#[cfg(test)]
pub fn create_and_train_predictor_model(dict: Dict, tokens_chain: Vec<String>, train: bool, device: &Device) -> Result<Mlp, candle_core::Error> {
    // Create Varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = Mlp::new(dict, varmap, vb)?;

    // Optimizer settings
    let epochs = 30;
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

        let input: Vec<Tensor> = input.iter().map(|token| Tensor::new(model.dict.get_token_cos_encoding(token), &device)).collect::<Result<Vec<Tensor>, candle_core::Error>>()?;

        let output = model.dict.get_word_index(output.as_str())?;

        let output = one_hot(Tensor::new(output, &device)?, model.dict.len(), 1.0 as f32, 0.0 as f32)?;

        inputs.push(input);
        targets.push(output);
    }

    // Training loop
    for epoch in 0..epochs {
        let mut outputs: Vec::<Tensor> = Vec::new();

        for (_index, input) in inputs.iter().enumerate() {
            let output = model.forward(&input)?;
            outputs.push(output);
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

#[cfg(test)]
pub fn get_device() -> Result<Device, candle_core::Error> {
    let device = Device::Cpu;
    return Ok(device);
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::token_utils::{tokenize, tokens_to_dict};

    use super::*;

    #[test]
    #[ignore]
    fn test_lstm_predictor_hello_world() -> Result<(), candle_core::Error> {
        let tokens = tokenize("hello world");

        let dict = tokens_to_dict(tokens.clone());

        let device = get_device()?;

        let mut model = create_and_train_predictor_model(dict, tokens, true, &device)?;

        assert_eq!(model.predict_next_token("hello ", &device)?, "world");

        Ok(())
    }

    #[test]
    #[ignore]
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
    #[ignore]
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
    #[ignore]
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
    #[ignore]
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
    #[ignore]
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
}
