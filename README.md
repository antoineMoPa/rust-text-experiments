# rust-text-experiment

A small experimental Rust project exploring tokenization and training a basic neural network on token embeddings. This project demonstrates how to preprocess text data, convert it into embeddings, and train a simple model to encode and decode vocabulary tokens.

## Features

- Tokenization: Splits input text into individual tokens, separating words and punctuation.
- Vocabulary generation: Builds a unique vocabulary from the tokens.
- Vocabulary dictionary: Converts vocabulary into a hash map with normalized float values for embeddings.
- Simple neural network training: Trains a minimal model to encode and decode tokens based on their embeddings.
- Tests.

## Requirements

- Rust
- The `nn` crate for neural network functionality

### Running Tests

This repo is meant to be used as a test playground for now, so just run tests:

```bash
cargo test
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

