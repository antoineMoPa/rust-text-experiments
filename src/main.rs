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

fn get_token_embedding(token: &str, dict: &std::collections::HashMap<String, f64>) -> Vec<f64> {
    let mut letter_embedding = 0.0;
    let value = *dict.get(token).unwrap();

    for letter in token.chars() {
        // simply cast letter to f64 and divide by 255
        let letter_value = letter as i32 as f64 / 10000.0;
        letter_embedding += letter_embedding + letter_value;
    }

    return vec![value, letter_embedding];
}

fn create_and_train_model_for_dict(dict: &std::collections::HashMap<String, f64>, embed_size: u32) -> NN {
    let mut model = create_model(2, embed_size);
    let mut examples = Vec::new();

    for (_i, token) in dict.iter().enumerate() {
        let token_embedding = get_token_embedding(token.0, dict);
        examples.push((token_embedding.clone(), token_embedding));
    }

    let mut trainer = model.train(&examples);
    trainer.halt_condition(nn::HaltCondition::Epochs(2000));
    trainer.go();

    return model;
}

fn main() {
    println!("Hello, world!");
}

fn are_embeddings_close(embedding1: Vec<f64>, embedding2: Vec<f64>, epsilon: f64) -> bool {
    let mut sum = 0.0;
    for i in 0..embedding1.len() {
        sum += (embedding1[i] - embedding2[i]).abs();
    }
    return sum < epsilon * embedding1.len() as f64;
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

        let model = create_and_train_model_for_dict(&dict, 2);

        let hello_embedding = get_token_embedding("Hello", &dict);

        assert!(are_embeddings_close(model.run(&*hello_embedding), hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(model.run(&*world_embedding), world_embedding, 0.15));
    }

    #[test]
    fn test_encode_larger_vocabulary() {
        let vocabulary = tokenize("This is a longer string, hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 10);

        let this_embedding = get_token_embedding("This", &dict);
        assert!(are_embeddings_close(model.run(&*this_embedding), this_embedding, 0.15));

        let is_embedding = get_token_embedding("is", &dict);
        assert!(are_embeddings_close(model.run(&*is_embedding), is_embedding, 0.15));

        let a_embedding = get_token_embedding("a", &dict);
        assert!(are_embeddings_close(model.run(&*a_embedding), a_embedding, 0.15));

        let longer_embedding = get_token_embedding("longer", &dict);
        assert!(are_embeddings_close(model.run(&*longer_embedding), longer_embedding, 0.15));

        let string_embedding = get_token_embedding("string", &dict);
        assert!(are_embeddings_close(model.run(&*string_embedding), string_embedding, 0.15));

        let comma_embedding = get_token_embedding(",", &dict);
        assert!(are_embeddings_close(model.run(&*comma_embedding), comma_embedding, 0.15));

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(are_embeddings_close(model.run(&*hello_embedding), hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(model.run(&*world_embedding), world_embedding, 0.15));

        let exclamation_embedding = get_token_embedding("!", &dict);
        assert!(are_embeddings_close(model.run(&*exclamation_embedding), exclamation_embedding, 0.15));
    }
}
