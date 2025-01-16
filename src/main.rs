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

fn create_vocabulary(tokens: Vec<String>) -> Vec<String> {
    let mut vocabulary = Vec::new();
    for token in tokens {
        if !vocabulary.contains(&token) {
            vocabulary.push(token);
        }
    }
    return vocabulary;
}


fn vocabulary_to_dict(vocabulary: Vec<String>) -> std::collections::HashMap<String, f64> {
    let mut vocabulary_dict = std::collections::HashMap::new();
    for (i, token) in vocabulary.iter().enumerate() {
        vocabulary_dict.insert(token.clone(), i as f64 / vocabulary.len() as f64);
    }
    return vocabulary_dict;
}

fn create_model(size: u32, embed_size: u32) -> NN {
    return NN::new(&[size, embed_size, size]);
}

fn create_and_train_model_for_dict(dict: &std::collections::HashMap<String, f64>, embed_size: u32) -> NN {
    let mut model = create_model(1, embed_size);
    let mut examples = Vec::new();

    for (_i, token) in dict.iter().enumerate() {
        let value = *token.1;
        examples.push((vec![value], vec![value]));
    }

    let mut trainer = model.train(&examples);
    trainer.halt_condition(nn::HaltCondition::Epochs(1000));
    trainer.go();

    return model;
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

    #[test]
    fn test_create_vocabulary() {
        let tokens = tokenize("Hello, world!");

        assert_eq!(create_vocabulary(tokens), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_vocabulary_to_dict() {
        let vocabulary = tokenize("Hello, world!");
        let vocabulary_dict = vocabulary_to_dict(vocabulary);

        assert_ne!(vocabulary_dict.get("Hello").unwrap(),
                   vocabulary_dict.get("world").unwrap());

        assert!(*vocabulary_dict.get("world").unwrap() > 0.0);
    }

    #[test]
    fn test_encode_vocabulary() {
        let vocabulary = tokenize("Hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 1);

        let hello = dict.get("Hello").unwrap();
        assert!((model.run(&[*hello])[0] - *hello).abs() < 0.15);

        let world = dict.get("world").unwrap();
        assert!((model.run(&[*world])[0] - *world).abs() < 0.15);
    }

    #[test]
    fn test_encode_larger_vocabulary() {
        let vocabulary = tokenize("This is a longer string, hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 1);

        let hello = dict.get("hello").unwrap();
        assert!((model.run(&[*hello])[0] - *hello).abs() < 0.1);

        let world = dict.get("world").unwrap();
        assert!((model.run(&[*world])[0] - *world).abs() < 0.1);

        let longer = dict.get("longer").unwrap();
        assert!((model.run(&[*longer])[0] - *longer).abs() < 0.1);
    }
}
