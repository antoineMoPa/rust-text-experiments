# rust-text-experiments

A small experimental Rust project exploring tokenization and training a basic neural network on token embeddings. This project demonstrates how to preprocess text data, convert it into embeddings, and train a simple model to encode and decode vocabulary tokens.

## Features

- A corpus of increasing difficulty (level_0, level_1, etc.)
- An encoder-decoder to build vector representation of words (with no meaning for now).
- 3 experimental models. Currently I'm iterating on the attention-based model.

## First model: classic neural network

[See doc here](./docs/simple_predictor.md)


## Second model: using LSTM

[See doc here](./docs/lstm_predictor.md)


## Current model: Attention

[See doc here](./docs/attention_predictor.md)

### Running

```bash
cargo run --release pretrain_encoder_decoder
cargo run --release train
cargo run --release self_test
cargo run --release run The cat sat on
# the mat.
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
