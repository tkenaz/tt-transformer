# Tensor-Train Transformer Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

Official implementation of **Layer-Specific Tensor-Train Compression for Transformers**.

## 🎯 Key Innovation

We demonstrate that different components of transformers have vastly different compression tolerance:
- **Attention layers**: Require conservative compression (~3x)
- **Feed-forward layers**: Can handle aggressive compression (~20x)

This insight leads to **4.5x better performance** compared to uniform compression approaches.

## 📊 Results

| Method | Compression | Perplexity | Quality |
|--------|------------|------------|---------|
| No Compression | 1.0x | ~35 | Baseline |
| **Uniform TT** | 2.2x | 198.92 | ❌ Poor |
| **Hybrid TT (Ours)** | 2.5x | 44.66 | ✅ Good |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/tensor-tt-compression.git
cd tensor-tt-compression
pip install -r requirements.txt
```

### Basic Usage

```python
from src.tt_transformer_hybrid import HybridTTTransformer, HybridTTConfig

# Create model with hybrid compression
config = HybridTTConfig(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    attention_ranks=[24, 48, 24],  # Conservative for attention
    ffn_ranks=[8, 16, 8],          # Aggressive for FFN
)

model = HybridTTTransformer(config)
print(f"Compression: {model.get_compression_stats()['total']['compression_ratio']:.2f}x")

# Use as regular PyTorch model
output = model(input_ids)
```

### Run Examples

```bash
# Basic usage demo
python examples/basic_usage.py

# Analyze different compression configurations
python examples/compression_analysis.py
```

## 📁 Project Structure

```
tensor-tt-compression/
├── src/
│   ├── tt_linear.py              # Core TT-Linear layer implementation
│   ├── tt_transformer.py         # Standard TT-Transformer
│   └── tt_transformer_hybrid.py  # Hybrid compression (our method)
├── examples/
│   ├── basic_usage.py           # Simple usage example
│   └── compression_analysis.py  # Compare compression ratios
├── tests/
│   └── test_tt_layers.py       # Unit tests
└── requirements.txt
```

## 🔬 Method Overview

### Tensor-Train Decomposition

We decompose weight matrices W ∈ ℝ^(m×n) into a sequence of smaller 3D tensors (cores):

```
W ≈ G₁ ×₁ G₂ ×₁ ... ×₁ Gₖ
```

### Hybrid Compression Strategy

Instead of using the same TT-ranks for all layers, we apply:
- **High ranks** for attention projections (Q, K, V, O)
- **Low ranks** for feed-forward networks (FFN)

This is based on our finding that FFN layers contain more redundancy and can be compressed more aggressively without significant quality loss.

## 📈 Experimental Results

Our experiments on WikiText-2 show:

1. **Uniform compression fails**: Using the same ranks everywhere leads to poor quality
2. **Hybrid compression succeeds**: Different ranks for different components maintains quality
3. **Optimal configuration**: 
   - Attention: ranks [24, 48, 24] (~3x compression)
   - FFN: ranks [8, 16, 8] (~20x compression)

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{tt-transformer-2024,
  title={Layer-Specific Tensor-Train Compression for Transformers},
  author={[Author Names]},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the Tensor-Train decomposition method
- Transformer architecture from "Attention is All You Need"

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email].

---

**Note**: This is a research implementation. Trained model weights and detailed training scripts are available upon request.