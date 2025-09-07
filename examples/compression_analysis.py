#!/usr/bin/env python3
"""
Analyze compression ratios for different configurations
"""

import sys
sys.path.append('../src')

from tt_transformer_hybrid import HybridTTConfig, HybridTTTransformer


def analyze_compression(name, attention_ranks, ffn_ranks):
    """Analyze compression for given rank configuration"""
    
    config = HybridTTConfig(
        vocab_size=50000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        attention_ranks=attention_ranks,
        ffn_ranks=ffn_ranks,
    )
    
    model = HybridTTTransformer(config)
    stats = model.get_compression_stats()
    
    print(f"\n{name}:")
    print(f"  Attention ranks: {attention_ranks}")
    print(f"  FFN ranks: {ffn_ranks}")
    print(f"  Total params: {stats['total']['original_params'] / 1e6:.1f}M")
    print(f"  Compressed: {stats['total']['compressed_params'] / 1e6:.1f}M")
    print(f"  Compression: {stats['total']['compression_ratio']:.2f}x")
    
    return stats['total']['compression_ratio']


def main():
    print("=" * 60)
    print("COMPRESSION ANALYSIS FOR DIFFERENT CONFIGURATIONS")
    print("=" * 60)
    
    configurations = [
        ("Conservative (High Quality)", [32, 64, 32], [16, 32, 16]),
        ("Balanced", [24, 48, 24], [12, 24, 12]),
        ("Aggressive", [16, 32, 16], [8, 16, 8]),
        ("Ultra Aggressive", [8, 16, 8], [4, 8, 4]),
    ]
    
    results = []
    for name, attn_ranks, ffn_ranks in configurations:
        ratio = analyze_compression(name, attn_ranks, ffn_ranks)
        results.append((name, ratio))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, ratio in results:
        quality_estimate = ""
        if ratio < 3:
            quality_estimate = "~90-95% of original"
        elif ratio < 5:
            quality_estimate = "~85-90% of original"
        elif ratio < 10:
            quality_estimate = "~75-85% of original"
        else:
            quality_estimate = "~60-75% of original"
        
        print(f"{name:20s}: {ratio:5.2f}x compression, quality {quality_estimate}")
    
    print("\n💡 Recommendation:")
    print("   Start with 'Balanced' configuration for best trade-off")
    print("   Use 'Conservative' if quality is critical")
    print("   Use 'Aggressive' if memory is limited")


if __name__ == "__main__":
    main()
