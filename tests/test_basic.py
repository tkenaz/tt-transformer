#!/usr/bin/env python3
"""
Basic tests for TT-Transformer
"""

import sys
import torch
import pytest
sys.path.append('../src')

from tt_linear import TTLinear
from tt_transformer_hybrid import HybridTTTransformer, HybridTTConfig


def test_tt_linear():
    """Test TT-Linear layer"""
    layer = TTLinear(
        in_features=256,
        out_features=512,
        tt_ranks=[4, 8, 4]
    )
    
    x = torch.randn(10, 256)
    y = layer(x)
    
    assert y.shape == (10, 512), f"Expected shape (10, 512), got {y.shape}"
    print("✅ TT-Linear test passed")


def test_hybrid_transformer():
    """Test Hybrid TT-Transformer"""
    config = HybridTTConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        attention_ranks=[8, 16, 8],
        ffn_ranks=[4, 8, 4]
    )
    
    model = HybridTTTransformer(config)
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    output = model(input_ids)
    
    assert 'logits' in output, "Output should contain 'logits'"
    assert output['logits'].shape == (2, 32, 1000), f"Unexpected output shape: {output['logits'].shape}"
    print("✅ Hybrid Transformer test passed")


def test_compression_ratio():
    """Test compression calculation"""
    config = HybridTTConfig(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=4,
        attention_ranks=[16, 32, 16],
        ffn_ranks=[8, 16, 8]
    )
    
    model = HybridTTTransformer(config)
    stats = model.get_compression_stats()
    
    assert stats['total']['compression_ratio'] > 1.0, "Compression ratio should be > 1"
    print(f"✅ Compression test passed: {stats['total']['compression_ratio']:.2f}x compression")


if __name__ == "__main__":
    test_tt_linear()
    test_hybrid_transformer()
    test_compression_ratio()
    print("\n✅ All tests passed!")
