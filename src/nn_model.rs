use nn::NN;

use crate::embedding_utils::get_token_embedding;

fn create_model(size: u32, embed_size: u32) -> NN {
    return NN::new(&[size, embed_size, size]);
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


#[cfg(test)]
mod tests {
    use crate::{token_utils::{vocabulary_to_dict, tokenize}, embedding_utils::are_embeddings_close};

    use super::*;

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
    fn test_encode_vocabulary() {
        let vocabulary = tokenize("Hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 2);

        let hello_embedding = get_token_embedding("Hello", &dict);

        assert!(are_embeddings_close(&model.run(&*hello_embedding), &hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(&model.run(&*world_embedding), &world_embedding, 0.15));
    }

    #[test]
    fn test_encode_larger_vocabulary() {
        let vocabulary = tokenize("This is a longer string, hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 10);

        let this_embedding = get_token_embedding("This", &dict);
        assert!(are_embeddings_close(&model.run(&*this_embedding), &this_embedding, 0.15));

        let is_embedding = get_token_embedding("is", &dict);
        assert!(are_embeddings_close(&model.run(&*is_embedding), &is_embedding, 0.15));

        let a_embedding = get_token_embedding("a", &dict);
        assert!(are_embeddings_close(&model.run(&*a_embedding), &a_embedding, 0.15));

        let longer_embedding = get_token_embedding("longer", &dict);
        assert!(are_embeddings_close(&model.run(&*longer_embedding), &longer_embedding, 0.15));

        let string_embedding = get_token_embedding("string", &dict);
        assert!(are_embeddings_close(&model.run(&*string_embedding), &string_embedding, 0.15));

        let comma_embedding = get_token_embedding(",", &dict);
        assert!(are_embeddings_close(&model.run(&*comma_embedding), &comma_embedding, 0.15));

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(are_embeddings_close(&model.run(&*hello_embedding), &hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(&model.run(&*world_embedding), &world_embedding, 0.15));

        let exclamation_embedding = get_token_embedding("!", &dict);
        assert!(are_embeddings_close(&model.run(&*exclamation_embedding), &exclamation_embedding, 0.15));

        // Different words should have different embeddings
        assert!(!are_embeddings_close(&model.run(&*this_embedding), &world_embedding, 0.15));
    }

    #[test]
    fn test_close_typos() {
        let vocabulary = tokenize("This this is a longer longee string, hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let model = create_and_train_model_for_dict(&dict, 10);

        let a = get_token_embedding("longer", &dict);
        let b = get_token_embedding("longee", &dict);
        assert!(are_embeddings_close(&model.run(&*a), &b, 0.1));

        let a = get_token_embedding("This", &dict);
        let b = get_token_embedding("this", &dict);
        assert!(are_embeddings_close(&model.run(&*a), &b, 0.1));

        let a = get_token_embedding("longer", &dict);
        let b = get_token_embedding("This", &dict);
        assert!(!are_embeddings_close(&model.run(&*a), &b, 0.1));

        let a = get_token_embedding("this", &dict);
        let b = get_token_embedding("longee", &dict);
        assert!(!are_embeddings_close(&model.run(&*a), &b, 0.1));
    }
}
