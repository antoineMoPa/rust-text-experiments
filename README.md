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
cargo run --release run "Last year, "
    Finished `release` profile [optimized] target(s) in 0.08s
     Running `target/release/rust-text-experiments run 'Last year, '`
Loading model
Vocab, Embedding Size, Context Window, Epochs, Hidden Size, Num blocks, Num att. heads, LR, Batch Size
18652, 108, 64, 32, 512, 2, 12, 0.001, 256
Completing: '"Last year,  "'
a story of animals where the smallest life had to warn us.

The story begins around 265 million, the next time. Think of it as a single, small, and the ocean has a few one of the first one of a time, the plant is the way the world in one evening. The air was thick with the soil of the air, the air changed with the scent of water from the water below. It was a single body, like a tiny propeller, and its babies in a fish, and it’s not just much like the food web of wood or a car.

The next few fruit grows down far away, and then it might be a new way to help look back to the same time as it contributes. It is the kind of more "food" because they need to find new things, even if it might be a little one in a day. The trees are the best way, but they can’t see them as they come. It’s like how it doesn’t eat, or even stay in a forest in a desert.

---

**Step 6: The Ocean Blocks of 1910s. The Big Picture a flower of the same creature on the flower, and the flower of a bee’s garden with one one. The first year, a creature with a flower, and it’s not all living things that have to visit the male flowers that are the male part of the ocean’s most fascinating group of animals on Earth.

6. **Food Chains**: Many animals grow on the ocean. They’re not just the energy they’re not just about the energy—the process they make a difference.

---

**The Sun’s Energy fixation**

To understand flowers, let’s make a simple part of the Earth’s surface. The Calvin cycle is the process by which the Calvin cycle (the Calvin cycle) is the primary consumers, which can be engineered to be eaten by plants.

4. **Climate change**: A temperate plant is the growth and nutrients that can thrive in the ocean. For instance, the Arctic tundra’s magnetic field for the "salmon" and deer" (the anemone century) is another layer of photosynthesis is transferred to the next generation. Think of it like the tiny whales or the original photosynthesis—it is called the ) of the largest animal world in the world. The clownfish convert a byproduct of the first major reactions into three main molecules. The energy is released with the Calvin cycle (around 255 ago), a process that split energy in the atmosphere and eventually split into the Calvin cycle. Without photosynthesis, the energy flows through food chains, and the energy would be more energy would eventually would become more efficiently, though the energy would be impossible.

The next time, the energy, emerged with the rise of the tree. Some, like the **tree spider (the algae) might form the tree in the water teeming with an ecosystem.
2. **Carbon Sequestration:** These are the light "food, though it’s not all species can survive. For example can be damp or wide, the "salmon" (which means-stop) to stay safe, which can be eaten by a species or an ecosystem could not produce water or an organism. But in the forest is that of it can be a single species called the **forest (the fungus might), it can survive in the soil or water can survive in the wild.

2. **Biodiversity**: These are small and small animals that help species get energy in the water. They’re like tiny, gilled (like a caterpillar), a fish farm with a gas that starts leaves!) can be eaten by other fish, while a small tree’s head eating the tree’s surface.
[...]
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
