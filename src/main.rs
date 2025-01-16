use nn::NN;

fn tokenize(input: &str) -> Vec<String> {
    let split_symbols = [' ', ',', '.', '!', '?', ';', ':', '\n', '\t'];


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

    return tokens;
}

fn create_model(size: u32, embed_size: u32) -> NN {
    return NN::new(&[size, embed_size, size]);
}

fn main() {
    println!("Hello, world!");
}

// test tokenize

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(tokenize("Hello, world!"), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_train_model() {
        let mut model = create_model(3, 1);
        let examples = [
            (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
            (vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]),
        ];
        let mut trainer = model.train(&examples);
        trainer.halt_condition(nn::HaltCondition::Epochs(1000));
        trainer.go();

        assert_eq!(model.run(&[0.0, 0.0, 0.0])[0].round(), 0.0);
        assert_eq!(model.run(&[1.0, 1.0, 1.0])[0].round(), 1.0);
    }
}
