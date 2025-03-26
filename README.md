# rust-text-experiment

A small experimental Rust project exploring tokenization and training a basic neural network on token embeddings. This project demonstrates how to preprocess text data, convert it into embeddings, and train a simple model to encode and decode vocabulary tokens.

## Features

- Tokenization: Splits input text into individual tokens, separating words and punctuation.
- Vocabulary generation: Builds a unique vocabulary from the tokens.
- Vocabulary dictionary: Converts vocabulary into a hash map with normalized float values for embeddings.
- Simple neural network training: Trains a minimal model to encode and decode tokens based on their embeddings.
- Tests.

## First model: classic neural network

[See doc here](./docs/simple_predictor.md)

## Second model: using LSTM

[See doc here](./docs/lstm_predictor.md)


## Third model: Attention

[See doc here](./docs/attention_predictor.md)

### Running

```bash
cargo run pretrain_encoder_decoder
cargo run pretrain
cargo run self_test
cargo run run The cat sat on
# the mat.
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
