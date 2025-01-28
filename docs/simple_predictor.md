### ✅ Classic Neural Net Text predictor

Neural net capable of outputting words based on the previous output.

**Final structure**

(Remember diagrams in AI are read from the bottom to the top for some unknown reason).

![simple preditcor architecture](./simple_predictor.png)

**Description**

The input consist of all previous tokens, the role of the network is to predict the next token.

When evaluating (running) the model, we append the last generated token to the input sequence and drop the last token that falls out of the token window.

**Features**

- Context window of 10 can be tweaked by changing a constant
- Same for hidden size
- Tests make sure the model can work with inputs ranging from 10 to 1000 tokens, which helps prototyping.

**Work notes**

- Encoding spaces (" ") as part of words helped reduce repetitions. Else, it was too easy for the model to optimize for every next token to be a space, because spaces are omnipresent in language.
- Used tanh instead of sigmoid because sigmoid is not implemented in candle in metal (apple silicon).
- Final tanh + softmax layers really helped the performance vs just the hidden layer output.

**Training corpus**

The model is trained on the wikipedia article about [horses](https://en.wikipedia.org/wiki/Horse). It just knows how to spit out horse stuff, mostly copied from wikipedia.

**Example output**

```
$ ./target/debug/rust-text-experiments run Horses
Loading model
Predicting next token for: '"Horses "'
and a to sleep, allowing them to quickly, Humans began domesticating horses around 4000 BCE, and their domestication is believed to have been widespread by 3000 BCE. Horses to the subspecies, and toed animal of today. Humans began domesticating horses around 4000 anatomy, and their domestication is believed to have been widespread by 3000 BCE. Horses used domesticated animal live related to by , domesticated flee from Horses to the BCE, single, colors, markings, breeds, locomotion, and behavior.

Horses are adapted to run, allowing them to quickly believed predators, and possess an excellent sense of balance and a strong fight.or flight response. Related to this believed to flee from predators by to wild, asexcellent ferus are a small multi,toed creature, Eohippus, into the large, single
toed animal of today. Humans began domesticating horses around 4000 BCE, and their domestication is believed to have been widespread by 3000 BCE. Horses family domestication is believed to have been widespread by 3000 BCE. Horses to the subspecies caballus are domesticated, although some domesticated populations live in the wild as feral,horses. These covering populations to describe are predators in describe, locomotion  The the caballus
 single to to of today. Humans began domesticating horses around to BCE, and their domestication is believed to have been widespread by 3000 BCE. Horses is the subspecies caballus are domesticated, although some domesticated populations live in the wild as feral horses. These as term extensive is used to describe horses that have never been domesticated. There is an,extensive, specialized vocabulary used to describe equine related concepts, covering everything from anatomy to life stages, size, colors, markings, breeds,.locomotion, and behavior.
```

Some sentences make sense. Others don't.

**Conclusion**

- It’s pretty good for such a simple architecture.
- It’s a classic neural network, no recurring parts, but previous tokens do give some context.
- I imagine it would be hard to scale to huge context windows because all of the input needs to be provided at once. In modern architectures, the input can be provided sequentially so it does not need to sit in RAM / VRAM.
- One-hot output encoding might not scale well to big dictionaries. Embedded output could be useful? Then we could reverse-search the output token using some vector database. Though I'm not sure if modern architectures bother doing this.

**Questions left in the open**

- Could I use this model or similar to predict stock values / other time series?
