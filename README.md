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
cargo run --release train
cargo run --release self_test
cargo run --release run The cat sat on
# the mat.
```

## Sample output

```
cargo run --release run "Last year, " | head -n 20
    Finished `release` profile [optimized] target(s) in 0.08s
     Running `target/release/rust-text-experiments run 'Last year, '`
Loading model
Vocab, Embedding Size, Context Window, Epochs, Hidden Size, Num blocks, Num att. heads, LR, Batch Size
18508, 108, 64, 20, 256, 2, 12, 0.01, 256
Completing: '"Last year,  "'
she heard one of the world was a part of the way she had never been written to the words. "This was not a kind of something that had been a stranger. It had a choice, not just a place that he had never been, but to be a memory that had never been before. But he had been a memory of the people, who had been the first time, not a relic of a one, as he had been a memory that been a part of the memory of the town.

He thought about the way he’s own life, he noticed the way he had been waiting, the way he had never never seen.

The memory of him, a place he had been a memory of it was a memory. And he had been a memory of his own—it felt like a memory—he told a memory of the town—he had never been a stranger.”

He stood up, his mind racing with the possibilities of his own own memory—it was a part of him. The way he had been his own idea that he could be a stranger.

He thought about the people to understand the world in the memory. It was a good man named Elias, who noticed a memory—he had been written. His breath like his breath, waiting for him. He was here in his life.”

Elias, his mind racing. *"They have said, his voice steady with a mix of pain and determination. “I love,” she said, his voice a mix of pain and hope. “I love you know.”*

Elias smiled as she walked to her phone. “I love. They can’t keep this. The weight of her own friends and the world in her hands, and the way to be a little weight of it.

The summer morning of her own life on a small bench, his mind racing. He had tried to be a small memory. It was a single accident—one of the first time. But he knew, it was not as he was holding its own, his mind racing and the memory.
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
