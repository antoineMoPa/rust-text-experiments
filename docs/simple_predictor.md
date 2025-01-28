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

- Encoding space as part of words helped reduce repetitions. Else it was too easy for the model to optimize for every next token to be a space, because spaces are omnipresent in language.
- Used tanh instead of sigmoid because sigmoid is not implemented in candle in metal (apple silicon)
- Final tanh + softmax layers really helped the performance vs just the hidden layer output.

**Conclusion**

- It’s pretty good for such a simple architecture.
- It’s a classic neural network, no recurring parts, but previous tokens do give some context.
- I imagine it would be hard to scale to huge context windows because all of the input needs to be provided.
- One-hot output encoding might not scale well to big dictionaries.

**Questions**

- Could I use this model or similar to predict stock values / other time series?
