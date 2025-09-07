"""
Hybrid TT-Transformer with layer-specific compression strategies
Optimized for maximum quality with ~5x compression

Key insight: Different layer types have different compression tolerance
- Attention: Critical for quality, minimal compression (2-3x)
- FFN: High redundancy, aggressive compression (10-30x)
- Embeddings: Moderate compression (5-10x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math

from tt_linear import TTLinear


@dataclass
class HybridTTConfig:
    """Configuration for Hybrid TT-Transformer with layer-specific ranks"""
    
    # Model dimensions
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # Hybrid TT configuration
    # Different ranks for different layer types
    attention_ranks: Optional[List[int]] = None  # e.g., [32, 64, 32] for 2x compression
    ffn_ranks: Optional[List[int]] = None       # e.g., [8, 16, 8] for 30x compression
    embed_ranks: Optional[List[int]] = None     # e.g., [16, 32, 16] for 7x compression
    
    # Layer-wise adaptive ranks (if enabled)
    use_adaptive_ranks: bool = False
    adaptive_rank_schedule: Optional[Dict[str, List[List[int]]]] = None
    
    # Compression targets (for auto rank selection)
    attention_compression_target: float = 2.0   # Conservative for attention
    ffn_compression_target: float = 15.0        # Aggressive for FFN
    embed_compression_target: float = 7.0       # Moderate for embeddings
    
    # Training configuration
    use_gradient_checkpointing: bool = False
    tie_embeddings: bool = True
    
    def __post_init__(self):
        """Set default ranks if not provided"""
        if self.attention_ranks is None:
            # Conservative compression for attention
            self.attention_ranks = [32, 64, 32]
            
        if self.ffn_ranks is None:
            # Aggressive compression for FFN
            self.ffn_ranks = [12, 24, 12]
            
        if self.embed_ranks is None:
            # Moderate compression for embeddings
            self.embed_ranks = [16, 32, 16]
            
        if self.use_adaptive_ranks and self.adaptive_rank_schedule is None:
            # Default adaptive schedule: early layers less compressed
            self.adaptive_rank_schedule = self._create_adaptive_schedule()
    
    def _create_adaptive_schedule(self) -> Dict[str, List[List[int]]]:
        """Create default adaptive rank schedule"""
        schedule = {
            'attention': [],
            'ffn': []
        }
        
        for layer_idx in range(self.n_layers):
            if layer_idx < self.n_layers // 3:
                # Early layers: minimal compression
                schedule['attention'].append([24, 48, 24])
                schedule['ffn'].append([16, 32, 16])
            elif layer_idx < 2 * self.n_layers // 3:
                # Middle layers: moderate compression
                schedule['attention'].append([16, 32, 16])
                schedule['ffn'].append([12, 24, 12])
            else:
                # Late layers: aggressive compression
                schedule['attention'].append([12, 24, 12])
                schedule['ffn'].append([8, 16, 8])
        
        return schedule
    
    def get_layer_ranks(self, layer_idx: int, layer_type: str) -> List[int]:
        """Get ranks for specific layer"""
        if not self.use_adaptive_ranks:
            if layer_type == 'attention':
                return self.attention_ranks
            elif layer_type == 'ffn':
                return self.ffn_ranks
            else:
                return self.embed_ranks
        else:
            return self.adaptive_rank_schedule[layer_type][layer_idx]


class HybridMultiHeadAttention(nn.Module):
    """Multi-head attention with hybrid TT compression"""
    
    def __init__(self, config: HybridTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        
        # Get ranks for this specific layer
        attn_ranks = config.get_layer_ranks(layer_idx, 'attention')
        
        # Use TT-Linear with layer-specific ranks
        self.q_proj = TTLinear(
            in_features=config.d_model,
            out_features=config.d_model,
            tt_ranks=attn_ranks,
            bias=True
        )
        
        self.k_proj = TTLinear(
            in_features=config.d_model,
            out_features=config.d_model,
            tt_ranks=attn_ranks,
            bias=True
        )
        
        self.v_proj = TTLinear(
            in_features=config.d_model,
            out_features=config.d_model,
            tt_ranks=attn_ranks,
            bias=True
        )
        
        self.out_proj = TTLinear(
            in_features=config.d_model,
            out_features=config.d_model,
            tt_ranks=attn_ranks,
            bias=True
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # Use -1e4 for fp16 compatibility
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class HybridFeedForward(nn.Module):
    """Feed-forward network with aggressive TT compression"""
    
    def __init__(self, config: HybridTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Get ranks for this specific layer
        ffn_ranks = config.get_layer_ranks(layer_idx, 'ffn')
        
        # FFN can be compressed more aggressively
        self.fc1 = TTLinear(
            in_features=config.d_model,
            out_features=config.d_ff,
            tt_ranks=ffn_ranks,
            bias=True
        )
        
        self.fc2 = TTLinear(
            in_features=config.d_ff,
            out_features=config.d_model,
            tt_ranks=ffn_ranks,
            bias=True
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class HybridTransformerBlock(nn.Module):
    """Transformer block with hybrid TT compression"""
    
    def __init__(self, config: HybridTTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention = HybridMultiHeadAttention(config, layer_idx)
        self.feed_forward = HybridFeedForward(config, layer_idx)
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x


class HybridTTTransformer(nn.Module):
    """Hybrid TT-Transformer with layer-specific compression strategies"""
    
    def __init__(self, config: HybridTTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings - use regular embedding for large vocab
        # TT-Linear for embeddings with 50k vocab is too memory intensive
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position embeddings (regular - they're small)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks with layer-specific compression
        self.blocks = nn.ModuleList([
            HybridTransformerBlock(config, layer_idx)
            for layer_idx in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection - use regular linear for large vocab
        if config.tie_embeddings:
            self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device)
            ).unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        if return_dict:
            return {'logits': logits}
        return logits
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Calculate compression statistics for the hybrid model"""
        stats = {
            'layer_stats': [],
            'total': {
                'original_params': 0,
                'compressed_params': 0,
                'compression_ratio': 0
            }
        }
        
        # Analyze each layer
        for idx, block in enumerate(self.blocks):
            layer_info = {
                'layer_idx': idx,
                'attention': {},
                'ffn': {}
            }
            
            # Attention stats
            for name, module in block.attention.named_modules():
                if isinstance(module, TTLinear):
                    orig = module.in_features * module.out_features
                    comp = sum(p.numel() for p in module.parameters())
                    layer_info['attention'][name] = {
                        'original': orig,
                        'compressed': comp,
                        'ratio': orig / comp
                    }
            
            # FFN stats
            for name, module in block.feed_forward.named_modules():
                if isinstance(module, TTLinear):
                    orig = module.in_features * module.out_features
                    comp = sum(p.numel() for p in module.parameters())
                    layer_info['ffn'][name] = {
                        'original': orig,
                        'compressed': comp,
                        'ratio': orig / comp
                    }
            
            stats['layer_stats'].append(layer_info)
        
        # Calculate totals
        for name, module in self.named_modules():
            if isinstance(module, TTLinear):
                stats['total']['original_params'] += module.in_features * module.out_features
                stats['total']['compressed_params'] += sum(p.numel() for p in module.parameters())
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                # Don't compress embeddings and regular linears
                param_count = sum(p.numel() for p in module.parameters())
                stats['total']['original_params'] += param_count
                stats['total']['compressed_params'] += param_count
        
        stats['total']['compression_ratio'] = (
            stats['total']['original_params'] / stats['total']['compressed_params']
            if stats['total']['compressed_params'] > 0 else 0
        )
        
        return stats


def create_hybrid_model(
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    use_adaptive: bool = False
) -> HybridTTTransformer:
    """Create a hybrid TT-Transformer with optimized compression"""
    
    config = HybridTTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        
        # Hybrid compression strategy
        attention_ranks=[32, 64, 32],  # ~2x compression for attention
        ffn_ranks=[12, 24, 12],        # ~13x compression for FFN
        embed_ranks=[16, 32, 16],      # ~7x compression for embeddings
        
        # Enable adaptive ranks if requested
        use_adaptive_ranks=use_adaptive,
        
        # Training optimizations
        use_gradient_checkpointing=True,
        tie_embeddings=False  # Disable for now with TT embeddings
    )
    
    return HybridTTTransformer(config)


if __name__ == "__main__":
    # Test the hybrid model
    model = create_hybrid_model(
        vocab_size=50257,
        d_model=768,
        n_heads=12,
        n_layers=12,
        use_adaptive=True
    )
    
    # Print compression stats
    stats = model.get_compression_stats()
    print(f"Total compression: {stats['total']['compression_ratio']:.2f}x")
    print(f"Original params: {stats['total']['original_params']:,}")
    print(f"Compressed params: {stats['total']['compressed_params']:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    output = model(input_ids)
    print(f"Output shape: {output['logits'].shape}")
    
    # Print per-layer stats
    print("\nPer-layer compression:")
    for layer_stat in stats['layer_stats'][:3]:  # Show first 3 layers
        print(f"Layer {layer_stat['layer_idx']}:")
        for module_type in ['attention', 'ffn']:
            if layer_stat[module_type]:
                total_ratio = sum(
                    m['ratio'] for m in layer_stat[module_type].values()
                ) / len(layer_stat[module_type])
                print(f"  {module_type}: {total_ratio:.1f}x compression")
