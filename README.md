## Conditional Poem Generator
## Author-Conditioned LSTM Language Model

## Project Overview

This project implements a **conditional generative model for poetry**, based on a **recurrent neural network with Long Short-Term Memory (LSTM)** units.  
The model is trained to generate poems **character by character**, while being **explicitly conditioned on the author’s identity**, enabling generation in the stylistic manner of different poets.

The key idea is that **each poem begins with the author’s name**, which is used as a conditioning signal that influences the entire generation process.

---

## Motivation

Poetic style is strongly dependent on the author.  
By conditioning the language model on the author identity, the system can:

- Learn stylistic patterns specific to individual poets
- Generate poems that resemble the vocabulary, rhythm, and structure of a given author
- Support controllable text generation without separate models per author

---

## Data Representation

Each poem in the dataset is represented as a pair:

**(author, character_sequence)**

The poem text is processed at **character level**, which allows:

- Fine-grained control over punctuation, rhythm, and formatting
- Avoidance of fixed word vocabularies
- Natural handling of rare or creative word forms

Special characters are introduced:

- **start character** `{` — marks the beginning of a poem  
- **end character** `}` — marks the end of a poem  
- **unknown character** `@` — for rare or unseen symbols  
- **padding character** `|` — used for batch processing  

---

## Model Architecture

The model is a **multi-layer LSTM language model**, extended with **author conditioning**.

Main components:

- **Character embedding layer**
- **Author embedding layers for initial hidden and cell states**
- **Stacked LSTM network**
- **Linear projection layer to character vocabulary**

The model predicts the **next character** given all previous characters and the author identity.

---

## Author Conditioning Mechanism

The author information is injected into the model **only once**, at the beginning of generation, by initializing the LSTM hidden states.

For each author, two learned embeddings are used:

- One for the **initial hidden state** \( h_0 \)
- One for the **initial cell state** \( c_0 \)

These embeddings are reshaped to match the number of LSTM layers:

- Shape: **(num_layers, batch_size, hidden_size)**

This design ensures that:

- The author identity influences the entire generation
- Style information is preserved over long sequences
- No explicit author tokens are required inside the poem text

---

## LSTM Language Modeling

At each time step, the LSTM processes one character embedding and updates its internal states.

The output hidden state is projected to the character vocabulary:

- Producing logits over all possible next characters
- Converted to probabilities via softmax

The model is trained using **cross-entropy loss**, comparing predicted characters to the ground-truth next characters.

Padding characters are ignored during loss computation.

---

## Training Procedure

Training is performed using:

- **Mini-batch training**
- **Packed sequences** to handle variable-length poems efficiently
- **Adam optimizer**
- **Dropout** for regularization

The training objective is to minimize the **negative log-likelihood** of the correct next character.

Model quality is evaluated using **perplexity** on a held-out test set.

---

## Text Generation

During generation:

1. The author name is provided as input
2. The LSTM hidden states are initialized using the author embeddings
3. A start character `{` is used as the seed
4. Characters are generated autoregressively until:
   - The end character `}` is produced, or
   - A maximum length is reached

---

## Temperature Sampling

A **temperature parameter** controls the randomness of generation:

- Low temperature → more conservative, repetitive text
- High temperature → more diverse and creative text

This allows balancing between stylistic fidelity and creativity.

---

## Advantages of the Approach

- Explicit and interpretable author conditioning
- Efficient reuse of a single model for multiple authors
- Character-level modeling captures poetic structure naturally
- Stable long-range generation using LSTM memory cells

---

## Summary

This project demonstrates a **conditional recurrent neural language model** capable of generating poetry in the style of specific authors.

It combines:

- LSTM-based sequence modeling
- Learned author embeddings
- Character-level generation
- Controlled text synthesis via conditioning

The system is well-suited for educational purposes, stylistic text generation, and experimentation with conditioned neural language models.