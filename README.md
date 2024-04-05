# Simple Transformer-based Language Model

This repository contains a simple implementation of a language model based on transformers using PyTorch. The model is trained on text data and can generate new text by continuing sequences of characters.

## Introduction

This project aims to provide a straightforward implementation of a language model using transformer architecture. It can be used for various text generation task.

## Code Description

This code implements a simple language modeling model based on the Transformer architecture using the PyTorch library. Below is a detailed description of each part of the code:

1. **Library Imports:**

   - `torch`: The PyTorch library for neural network operations.
   - `torch.nn`: PyTorch module for building neural networks.
   - `torch.nn.functional`: PyTorch module containing activation functions and loss functions.
2. **Hyperparameters:**

   - `batch_size`: The batch size for training data.
   - `block_size`: The maximum context length for predictions.
   - `max_iters`: The maximum number of training iterations.
   - `eval_interval`: Interval for evaluating losses on training and validation sets.
   - `learning_rate`: Learning rate for optimization.
   - `device`: The device on which the training occurs (GPU or CPU).
   - `eval_iters`: Number of iterations for evaluating losses on each epoch.
   - `n_embd`: Embedding size.
   - `n_head`: Number of attention heads in multi-head attention.
   - `n_layer`: Number of layers in the Transformer.
   - `dropout`: Dropout coefficient.
3. **Random Seed Initialization:**

   - Sets a random initial state for reproducibility of results.
4. **Loading Text Data:**

   - Opens the `input.txt` file containing text for model training.
   - Unique characters are extracted from the text, and dictionaries for converting characters to integers and vice versa are created.
5. **Splitting Data into Training and Validation Sets:**

   - Converts the text into numerical format and splits it into training and validation sets in a 90% to 10% ratio.
6. **Data Loading:**

   - Defines the `get_batch(split)` function, which generates a small batch of data for training or validation.
7. **Loss Estimation:**

   - Defines the `estimate_loss()` function, which estimates the model's losses on the training and validation sets.
8. **Definition of the Head Class:**

   - The `Head` class represents one head of self-attention in the Transformer.
9. **Definition of the MultiHeadAttention Class:**

   - The `MultiHeadAttention` class represents multiple heads of self-attention in the Transformer.
10. **Definition of the FeedForward Class:**

    - The `FeedForward` class represents a simple linear layer followed by a non-linear activation function.
11. **Definition of the Block Class:**

    - The `Block` class represents a Transformer block, which consists of a multi-head attention layer and a feedforward layer.
12. **Definition of the BigramLanguageModel Class:**

    - The `BigramLanguageModel` class represents the language modeling model based on the Transformer architecture.
    - It includes layers for token embeddings, positional embeddings, a sequence of Transformer blocks, and layers for normalization and prediction.

## How to Use

1. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

2. Run the `bigram.py` script to start training the model:

```bash
python bigram.py
```

## Training Results

The model was trained on an NVIDIA T4 GPU for 1 hour. Here are the training results:

- Initial training loss: 4.4360
- Final training loss: 1.0125
- Initial validation loss: 4.4359
- Final validation loss: 1.6178

## Generated Text Example

After training, the model generated the following text:

```
VINGBROKE:
My royal throw, what myself stand up London,
Selves that virtue a strange first and fallseK's flies,
I have statutes of my cousin: 'tis not breast
Her faith, hast that follows in hell.

HENRY BOLINGBROKE:
One times are rid, my lord, unforth!
Be poison, of all the man, hear me not word him at
will processors the wive and kingly spreed!
How long! sour my serve lived, to my sweet.

KING RICHARD II:
And thus.!

LADY CARENCE:

YORK:
Why, lie let my mother; but I would not
Was newly allow'd
```


## License

This project is inspired by the lecture "Let’s build GPT: from scratch, in code, spelled out" by Andrej Karpathy. You can watch the lecture [here](https://youtu.be/kCc8FmEb1nY?si=zWxROR-gMJb9ubLX).
