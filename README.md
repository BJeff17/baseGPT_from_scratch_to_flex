
# baseGPT_from_scratch_to_flex

**baseGPT_from_scratch_to_flex** is a minimal GPT-like language model implemented entirely from scratch using PyTorch.
The project focuses on understanding every core component of a transformer-based language model, from tokenization to attention and training.

The goal is not performance, but **clarity, control, and flexibility**.

---

## Project Goals

* Implement a GPT-style transformer **from first principles**
* Understand attention, masking, and multi-head composition
* Build a custom tokenizer (BPE-like) without external libraries
* Train a causal language model on raw text
* Keep the code readable and easily modifiable

---

## Model Architecture

The model follows a classic decoder-only transformer design.

### Components

* **Token Embedding**
* **Sinusoidal Positional Encoding**
* **Masked Self-Attention**
* **Multi-Head Attention**
* **Feed-Forward Network**
* **Residual Connections + LayerNorm**
* **Linear Language Modeling Head**

All components are implemented manually using PyTorch primitives.

---

## Transformer Block

Each block contains:

1. Masked multi-head self-attention
2. Residual connection + LayerNorm
3. Feed-forward network
4. Residual connection + LayerNorm

The model can be scaled by adjusting:

* number of blocks
* number of heads
* embedding size
* context length

---

## Tokenizer

A custom **byte-level BPE-inspired tokenizer** is included.

### Features

* Starts from raw UTF-8 bytes
* Iteratively merges most frequent byte pairs
* Fully reversible encoding / decoding
* No dependency on external tokenization libraries

This allows training on arbitrary text while preserving full control over vocabulary construction.

---

## Dataset

Text is converted into training samples using a **sliding window** strategy:

* Fixed context length
* Configurable stride
* Automatic padding
* Produces `(input, target)` token pairs

This enables causal language modeling from continuous text streams.

---

## Training

* Optimizer: Adam
* Loss: Cross-Entropy
* Objective: next-token prediction
* Fully autoregressive (causal masking)

The training loop is intentionally minimal to keep behavior transparent.

## Architecture Diagram

```
Input Tokens (B, T)
        │
        ▼
┌──────────────────────┐
│ Token Embedding      │  Embedding(vocab_size → n_embed)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Positional Encoding  │  Sinusoidal
└──────────────────────┘
        │
        ▼
   (Token + Position)
        │
        ▼
┌────────────────────────────────────────────┐
│            Transformer Blocks × N          │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │ Masked Multi-Head Self-Attention     │  │
│  │  - Q, K, V projections               │  │
│  │  - Causal mask (no future tokens)    │  │
│  │  - Softmax attention                 │  │
│  └──────────────────────────────────────┘  │
│                │                           │
│                ▼                           │
│        Residual + LayerNorm                │
│                │                           │
│                ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │ Feed-Forward Network                 │  │
│  │  Linear → ReLU → Linear              │  │
│  └──────────────────────────────────────┘  │
│                │                           │
│                ▼                           │
│        Residual + LayerNorm                │
│                                            │
└────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Linear Output Head   │  (n_embed → vocab_size)
└──────────────────────┘
        │
        ▼
Next-Token Logits (B, T, vocab_size)
```

---

## Notes on Design Choices

* **Causal masking** ensures autoregressive behavior
* **Residual connections** stabilize training
* **LayerNorm** is applied after attention and feed-forward blocks
* **Tokenizer is byte-level**, enabling full UTF-8 coverage
* **Everything is modular**, making experimentation easy

---
> This architecture mirrors a decoder-only transformer (GPT-style), implemented from scratch with explicit control over every component.
---

## Usage

### 1. Train the tokenizer

```python
tokenizer = Tokenizer(vocab_size=500)
tokenizer.trainTokenizer(text)
```

### 2. Build dataset

```python
dataset = SlidingWindowDataset(
    tokenizer=tokenizer,
    n_context=64,
    texts=[text]
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. Train the model

```python
model = B2lM(
    vocab_size=len(tokenizer),
    n_embed=128,
    n_block=4,
    n_k=32,
    n_v=32,
    n_head=4,
    n_ff=256
)

model.train(loader, epochs=10)
```

---

## Why This Project Exists

Most GPT implementations hide complexity behind frameworks.
This project removes those abstractions to expose:

* how attention really works,
* how masking enforces causality,
* how tokenization affects learning,
* how architectural choices change behavior.

---

## Limitations

* Not optimized for speed or scale
* No KV caching
* No mixed precision
* No sampling utilities (yet)
* Intended for learning, not production

---

## Planned Extensions

* Text generation utilities
* KV cache for inference
* Layer-wise configuration
* Better tokenizer serialization
* Training on larger corpora
* Weight tying

---

## Dependencies

```bash
pip install torch tqdm
```

Python 3.9+ recommended.

---

## Contributing

Contributions are welcome, especially if they:

* improve clarity without hiding logic,
* add educational value,
* extend flexibility without adding heavy dependencies.

Feel free to open issues or pull requests.

---

## License

Open for experimentation and learning.

---
