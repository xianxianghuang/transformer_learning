# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CBOW (Continuous Bag of Words) word embedding model implemented in PyTorch. The model learns dense vector representations of words by predicting a target word from its context words.

## Project Structure

```
CBOW/
├── data/ptb.txt              # Training corpus (Penn Treebank)
├── src/
│   ├── data_loader.py        # Corpus preprocessing, vocabulary building
│   ├── model.py              # CBOW model with negative sampling
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation (similarity, analogy, visualization)
│   └── utils.py              # Utility functions
├── checkpoints/              # Saved models
├── requirements.txt          # Dependencies
├── SPEC.md                   # Specification document
└── README.md                 # Documentation
```

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Requirements: torch>=1.9.0, numpy>=1.19.0, scikit-learn>=0.24.0, matplotlib>=3.3.0
```

## Common Commands

```bash
# Train model (with recommended parameters)
cd src && python train.py

# Train with PTB + WikiText-2
python train.py --data_path "data/ptb.train.txt,data/ptb.valid.txt,data/wiki.train.tokens,data/wiki.valid.tokens"

# Evaluate (similarity & analogy)
python evaluate.py --model_path ../checkpoints/cbow_final.pt

# Comprehensive evaluation (similarity, analogy, clustering, visualization)
python comprehensive_eval.py --model_path ../checkpoints/cbow_final.pt --visualize
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding_dim` | 300 | Word vector dimension |
| `--window_size` | 5 | Context window radius |
| `--learning_rate` | **10.0** | Learning rate (important: use large value) |
| `--epochs` | **100** | Training epochs (recommended: 100) |
| `--batch_size` | 512 | Batch size |
| `--negative_samples` | 5 | Negative sampling count |
| `--min_count` | 5 | Minimum word frequency |

## Architecture

- **Input**: Context word indices (batch_size, window_size*2)
- **Embedding Layer**: Maps word indices to dense vectors
- **Mean Pooling**: Averages context word embeddings
- **Output**: Negative sampling for efficient training

## Data Format

The corpus expects plain text file at `data/ptb.txt` with whitespace-separated words. The demo corpus contains ~400 words with semantic relationships for testing.

## Evaluation Features

- Word similarity (cosine similarity)
- Word analogy tasks (e.g., "man:king :: woman:?")
- t-SNE visualization
