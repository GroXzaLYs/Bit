# BIT - Production Ready Transformer

BIT is a production-ready, localized transformer model built from scratch. This refactor introduces a BPE tokenizer, KV-cache optimized inference, disk-streaming for large datasets, and a FastAPI serving layer.

## Features
- **Tokenizer**: BPE (Byte Pair Encoding) with a vocabulary size of 10,000.
- **Architecture**: Transformer with Multi-Head Attention and SwiGLU.
- **Inference**: Optimized with KV-caching (10x faster) and Top-k/Top-p sampling.
- **Training**: Early stopping, cosine learning rate annealing, and Tensorboard logging.
- **Serving**: FastAPI-based REST API for production deployment.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py master_dataset.txt
```

### Chat
```bash
python chat.py --model bit_model.pt
```

### API Server
```bash
python -m bit.serve
```

## Documentation
- [Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
