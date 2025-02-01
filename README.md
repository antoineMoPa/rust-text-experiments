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


## Example output

Here is a sample of the model trained on the wikipedia "Horse" article.

```
Predicting next token for: '"Horses "'
subspecies the subspecies caballus  The horse (Equus ferus caballus) is a domesticated, one-toed, hoofed mammal. It belongs to the taxonomic family Equidae and is one of two believed the of two two family the of is one domesticated wild the excellent wild horses horses 4000 BCE. and in the subspecies caballus are domesticated, although some domesticated populations live in the wild horses subspecies horses are predators to sleep wild is concepts to the taxonomic family Equidae and is one of two never subspecies of Equus ferus. The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today. Humans began domesticating horses around 4000 BCE, and their domestication is believed to have been widespread by 3000 BCE. Horses in the subspecies caballus are domesticated, although some domesticated populations live in the wild horses to the trait        The horse (Equus ferus caballus) is a domesticated, one-toed, hoofed mammal
```

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
