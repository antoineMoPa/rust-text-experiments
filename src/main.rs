mod embedding_utils;
mod token_utils;
mod nn_model;
mod candle_model;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::Device;
    use token_utils::{
        tokenize,
        vocabulary_to_dict,
    };

    use embedding_utils::{
        get_token_embedding,
        are_embeddings_close,
    };

    #[test]
    fn test_candle_encoder_decoder_hello_world() -> Result<(), candle_core::Error> {
        let vocabulary = tokenize("hello, world!");
        let dict = vocabulary_to_dict(vocabulary);

        let device = Device::Cpu;
        let model = candle_model::create_and_train_model_for_dict(&dict, 2).unwrap();

        let hello_embedding = get_token_embedding("hello", &dict);
        assert!(are_embeddings_close(&model.run(&hello_embedding, &device)?, &hello_embedding, 0.15));

        let world_embedding = get_token_embedding("world", &dict);
        assert!(are_embeddings_close(&model.run(&world_embedding, &device)?, &world_embedding, 0.15));

        // Different words should have different results
        assert!(!are_embeddings_close(&model.run(&hello_embedding, &device)?, &world_embedding, 0.15));

        Ok(())
    }
}
