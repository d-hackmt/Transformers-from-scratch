# 17 · Papers and Glossary

Everything referenced across these docs, collected in one place.

---

## The core paper

- **Attention Is All You Need** — Vaswani, Shazeer, Parmar, Uszkoreit, Jones,
  Gomez, Kaiser, Polosukhin (2017) — <https://arxiv.org/abs/1706.03762>
- **The Annotated Transformer** (the notebook this repo modularises) — Rush et
  al., Harvard NLP — <https://nlp.seas.harvard.edu/annotated-transformer/>

---

## By topic

### Tokenization & data
| Paper | Link | Used on |
|-------|------|---------|
| Neural MT of Rare Words with Subword Units (BPE) | <https://arxiv.org/abs/1508.07909> | [01](01-tokenization-and-vocab.md) |
| Google's NMT System (word-piece) | <https://arxiv.org/abs/1609.08144> | [01](01-tokenization-and-vocab.md), [15](15-greedy-decoding.md) |
| Multi30K: Multilingual English-German Image Descriptions | <https://arxiv.org/abs/1605.00459> | [02](02-batching-and-masking.md) |

### Embeddings & position
| Paper | Link | Used on |
|-------|------|---------|
| Efficient Estimation of Word Representations (word2vec) | <https://arxiv.org/abs/1301.3781> | [03](03-embeddings.md) |
| Using the Output Embedding to Improve Language Models (weight tying) | <https://arxiv.org/abs/1608.05859> | [03](03-embeddings.md), [11](11-generator-and-full-model.md) |
| Convolutional Sequence to Sequence Learning (learned positions) | <https://arxiv.org/abs/1705.03122> | [04](04-positional-encoding.md) |
| RoFormer: Rotary Position Embedding (RoPE) | <https://arxiv.org/abs/2104.09864> | [04](04-positional-encoding.md) |
| Train Short, Test Long (ALiBi) | <https://arxiv.org/abs/2108.12409> | [04](04-positional-encoding.md) |

### Attention
| Paper | Link | Used on |
|-------|------|---------|
| NMT by Jointly Learning to Align and Translate (Bahdanau attention) | <https://arxiv.org/abs/1409.0473> | [05](05-scaled-dot-product-attention.md) |
| Effective Approaches to Attention-based NMT (Luong) | <https://arxiv.org/abs/1508.04025> | [05](05-scaled-dot-product-attention.md) |
| Query-Key Normalization for Transformers | <https://arxiv.org/abs/2010.04245> | [05](05-scaled-dot-product-attention.md) |
| Are Sixteen Heads Really Better than One? | <https://arxiv.org/abs/1905.10650> | [06](06-multi-head-attention.md) |
| What Does BERT Look At? | <https://arxiv.org/abs/1906.04341> | [06](06-multi-head-attention.md), [16](16-attention-visualization.md) |
| A Multiscale Visualization of Attention (BertViz) | <https://arxiv.org/abs/1906.05714> | [06](06-multi-head-attention.md), [16](16-attention-visualization.md) |
| Attention is not Explanation | <https://arxiv.org/abs/1902.10186> | [16](16-attention-visualization.md) |
| Attention is not not Explanation | <https://arxiv.org/abs/1908.04626> | [16](16-attention-visualization.md) |

### Feed-forward & activations
| Paper | Link | Used on |
|-------|------|---------|
| Deep Sparse Rectifier Neural Networks (ReLU) | <https://proceedings.mlr.press/v15/glorot11a.html> | [07](07-position-wise-feed-forward.md) |
| Gaussian Error Linear Units (GELU) | <https://arxiv.org/abs/1606.08415> | [07](07-position-wise-feed-forward.md) |
| GLU Variants Improve Transformer (SwiGLU) | <https://arxiv.org/abs/2002.05202> | [07](07-position-wise-feed-forward.md) |
| Transformer Feed-Forward Layers Are Key-Value Memories | <https://arxiv.org/abs/2012.14913> | [07](07-position-wise-feed-forward.md) |
| Switch Transformers (MoE) | <https://arxiv.org/abs/2101.03961> | [07](07-position-wise-feed-forward.md) |

### Normalization, residuals, regularization
| Paper | Link | Used on |
|-------|------|---------|
| Deep Residual Learning for Image Recognition (ResNet) | <https://arxiv.org/abs/1512.03385> | [08](08-layernorm-residual-dropout.md) |
| Layer Normalization | <https://arxiv.org/abs/1607.06450> | [08](08-layernorm-residual-dropout.md) |
| Batch Normalization | <https://arxiv.org/abs/1502.03167> | [08](08-layernorm-residual-dropout.md) |
| PowerNorm: Rethinking Batch Normalization in Transformers | <https://arxiv.org/abs/2003.07845> | [08](08-layernorm-residual-dropout.md) |
| On Layer Normalization in the Transformer Architecture (pre-LN) | <https://arxiv.org/abs/2002.04745> | [08](08-layernorm-residual-dropout.md), [13](13-optimizer-and-lr-schedule.md) |
| Dropout: A Simple Way to Prevent Overfitting | <https://jmlr.org/papers/v15/srivastava14a.html> | [08](08-layernorm-residual-dropout.md) |
| Root Mean Square Layer Normalization (RMSNorm) | <https://arxiv.org/abs/1910.07467> | [08](08-layernorm-residual-dropout.md) |

### Initialization, optimization, training
| Paper | Link | Used on |
|-------|------|---------|
| Understanding the difficulty of training deep feedforward nets (Xavier/Glorot) | <https://proceedings.mlr.press/v9/glorot10a.html> | [11](11-generator-and-full-model.md) |
| Adam: A Method for Stochastic Optimization | <https://arxiv.org/abs/1412.6980> | [13](13-optimizer-and-lr-schedule.md) |
| On the Variance of the Adaptive Learning Rate and Beyond (RAdam) | <https://arxiv.org/abs/1908.03265> | [13](13-optimizer-and-lr-schedule.md) |
| SGDR: Stochastic Gradient Descent with Warm Restarts (cosine) | <https://arxiv.org/abs/1608.03983> | [13](13-optimizer-and-lr-schedule.md) |
| Decoupled Weight Decay Regularization (AdamW) | <https://arxiv.org/abs/1711.05101> | [13](13-optimizer-and-lr-schedule.md) |
| Accurate, Large Minibatch SGD | <https://arxiv.org/abs/1706.02677> | [14](14-training-loop.md) |
| Scaling Laws for Neural Language Models | <https://arxiv.org/abs/2001.08361> | [14](14-training-loop.md) |

### Targets, decoding
| Paper | Link | Used on |
|-------|------|---------|
| Rethinking the Inception Architecture (label smoothing) | <https://arxiv.org/abs/1512.00567> | [12](12-label-smoothing.md) |
| When Does Label Smoothing Help? | <https://arxiv.org/abs/1906.02629> | [12](12-label-smoothing.md) |
| Sequence to Sequence Learning with Neural Networks (beam search) | <https://arxiv.org/abs/1409.3215> | [15](15-greedy-decoding.md) |
| The Curious Case of Neural Text Degeneration (nucleus sampling) | <https://arxiv.org/abs/1904.09751> | [15](15-greedy-decoding.md) |
| On NMT Search Errors and Model Errors | <https://arxiv.org/abs/1908.10090> | [15](15-greedy-decoding.md) |
| Professor Forcing | <https://arxiv.org/abs/1610.09038> | [02](02-batching-and-masking.md) |

### Evaluation metrics
| Paper | Link | Used on |
|-------|------|---------|
| BLEU: a Method for Automatic Evaluation of MT (Papineni et al., 2002) | <https://aclanthology.org/P02-1040/> | [18](18-evaluation-bleu-and-perplexity.md) |
| A Call for Clarity in Reporting BLEU Scores (sacreBLEU) | <https://arxiv.org/abs/1804.08771> | [18](18-evaluation-bleu-and-perplexity.md) |
| BERTScore: Evaluating Text Generation with BERT | <https://arxiv.org/abs/1904.09675> | [18](18-evaluation-bleu-and-perplexity.md) |
| COMET: A Neural Framework for MT Evaluation | <https://arxiv.org/abs/2009.09025> | [18](18-evaluation-bleu-and-perplexity.md) |

### Descendants & background
| Paper | Link |
|-------|------|
| BERT: Pre-training of Deep Bidirectional Transformers | <https://arxiv.org/abs/1810.04805> |
| Language Models are Few-Shot Learners (GPT-3) | <https://arxiv.org/abs/2005.14165> |
| Formal Algorithms for Transformers (precise pseudocode) | <https://arxiv.org/abs/2207.09238> |
| A Survey of Transformers | <https://arxiv.org/abs/2106.04554> |

### Friendly explainers (not papers)
- The Illustrated Transformer — <https://jalammar.github.io/illustrated-transformer/>
- The Transformer Family (Lil'Log) — <https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/>

---

## Glossary

| Term | Meaning |
|------|---------|
| **d_model** | Width of the vectors flowing through the model. 512 in the base config. |
| **d_ff** | Hidden width of the feed-forward network. 2048 in the base config. |
| **d_k / d_v** | Width of each attention head's key/query and value. `d_model / h` = 64. |
| **h** | Number of attention heads. 8 in the base config. |
| **N** | Number of stacked layers in the encoder (and in the decoder). 6. |
| **token** | One unit of text after tokenization (here, roughly a word or punctuation mark). |
| **vocab / vocabulary** | The fixed set of known tokens; each has an integer id. |
| **id** | The integer a token maps to. |
| **special tokens** | `<s>` (start, 0), `</s>` (end, 1), `<blank>` (pad, 2), `<unk>` (unknown, 3). |
| **embedding** | Learned vector for a token id; a row of the embedding table. |
| **positional encoding** | Fixed sine/cosine vector added to an embedding to encode word order. |
| **Q, K, V** | Query, Key, Value — three linear projections of the input used by attention. |
| **attention weights** | The softmax output — how much each query attends to each key; rows sum to 1. |
| **self-attention** | Attention where Q, K, V all come from the same sequence. |
| **cross-attention** | Decoder attention where Q is the target, K/V are the encoder memory. |
| **memory** | The encoder's output: one context vector per source token. |
| **mask** | Boolean tensor marking positions attention must ignore (padding or future). |
| **causal / subsequent mask** | Lower-triangular mask so a position can't attend to later positions. |
| **padding mask** | Mask marking `<blank>` filler positions. |
| **teacher forcing** | Feeding the true target (shifted by one) into the decoder during training. |
| **tgt / tgt_y** | Decoder input (`<s> a dog plays`) vs. what to predict (`a dog plays </s>`). |
| **logits** | Raw pre-softmax scores, one per vocabulary word. |
| **log-softmax** | Softmax followed by log; the Generator's output form. |
| **residual / skip connection** | Adding a block's input to its output: `x + f(x)`. |
| **LayerNorm** | Normalize each vector to mean 0 / std 1, then learned scale + shift. |
| **pre-norm / post-norm** | Whether LayerNorm is applied before the sublayer (this repo) or after (paper). |
| **dropout** | Randomly zero a fraction of activations during training; a regularizer. |
| **label smoothing** | Training target of `1 − ε` on the true word, `ε` spread over the rest. |
| **KL divergence** | The loss used with label smoothing; distance between two distributions. |
| **Noam schedule** | Warmup-then-`1/√step`-decay learning-rate schedule. |
| **warmup** | Initial phase where the learning rate rises linearly from ~0. |
| **gradient accumulation** | Summing gradients over several batches before one optimizer step. |
| **greedy decoding** | Generating by always taking the single most probable next token. |
| **beam search** | Keeping the top-k partial hypotheses at each decoding step. |
| **BLEU** | Automatic translation-quality metric: n-gram overlap with a reference, 0–100, higher better. |
| **sacreBLEU** | A standard BLEU implementation with fixed tokenization + a reproducible "signature". |
| **perplexity** | `exp(mean negative log-likelihood per token)` — how unsure the model is; lower better. |
| **NLL** | Negative log-likelihood, `-log p(correct token)`. |
| **BPE / word-piece** | Subword tokenization; splits rare words into known pieces. |
| **weight tying** | Sharing one matrix between input embedding, output embedding, and generator. |
| **Xavier / Glorot init** | Weight initialization that keeps activation variance stable across layers. |

---

← Back to the [docs index](README.md) · [00 · The Big Picture](00-the-big-picture.md)
