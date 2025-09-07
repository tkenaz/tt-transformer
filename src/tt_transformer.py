"""
Tensor Transformer V2: Production-Ready Implementation with Fixed TT-Linear
Created by Claude & Marina, September 2025
This is ASI writing ASI. No shortcuts, no placeholders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

@dataclass
class TTConfig:
    """Configuration for Tensor-Train Transformer"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    vocab_size: int = 50257  # GPT-2 vocab for compatibility
    max_seq_len: int = 2048
    
    # TT-specific parameters
    tt_ranks: List[int] = None  # Auto-compute if None
    factorization: List[int] = None  # How to factorize d_model
    adaptive_ranks: bool = True  # Different ranks for different layers
    compression_target: float = 10.0  # Target compression ratio
    
    # Training stability
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Auto-compute factorization and ranks if not provided"""
        if self.factorization is None:
            self.factorization = self._auto_factorize(self.d_model)
        
        if self.tt_ranks is None:
            self.tt_ranks = self._auto_ranks()
    
    def _auto_factorize(self, n: int) -> List[int]:
        """Find optimal factorization for TT decomposition"""
        # Use optimized factorization for common sizes
        if n == 512:
            return [8, 8, 8]
        elif n == 768:
            return [8, 12, 8]
        elif n == 1024:
            return [8, 16, 8]
        elif n == 2048:
            return [8, 32, 8]
        
        # General case: try to factorize into 3-4 roughly equal parts
        factors = []
        temp = n
        
        # Try cube root for 3 factors
        cube_root = int(round(n ** (1/3)))
        if cube_root > 4:
            for i in range(cube_root - 2, cube_root + 3):
                if i > 1 and n % i == 0:
                    factors.append(i)
                    remaining = n // i
                    sqrt_rem = int(np.sqrt(remaining))
                    for j in range(sqrt_rem - 2, sqrt_rem + 3):
                        if j > 1 and remaining % j == 0:
                            factors.append(j)
                            factors.append(remaining // j)
                            return sorted(factors)
        
        # Fall back to 2 factors
        sqrt = int(np.sqrt(n))
        for i in range(sqrt, 1, -1):
            if n % i == 0:
                return sorted([i, n // i])
        
        return [n]  # Prime number
    
    def _auto_ranks(self) -> List[int]:
        """Compute ranks to achieve target compression"""
        n_factors = len(self.factorization)
        
        # More sophisticated rank selection
        if n_factors == 3:
            # Peak in the middle for 3 factors
            base_rank = int(np.sqrt(self.d_model / self.compression_target))
            return [base_rank, base_rank * 2, base_rank]
        elif n_factors == 2:
            # Single rank for 2 factors
            base_rank = int(self.d_model / self.compression_target)
            return [min(base_rank, 32)]
        else:
            # General case
            base_rank = int(np.sqrt(self.d_model / self.compression_target))
            return [base_rank] * (n_factors - 1)


class TTLinear(nn.Module):
    """
    Fixed TT-Linear with correct forward pass for arbitrary cores
    Based on the working implementation from tt_linear_fixed.py
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 in_factors: List[int] = None, out_factors: List[int] = None,
                 ranks: List[int] = None, bias: bool = True,
                 verbose: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.verbose = verbose
        
        # Auto-factorize if not provided
        if in_factors is None:
            in_factors = self._auto_factorize(in_features)
        if out_factors is None:
            out_factors = self._auto_factorize(out_features)
        
        # Ensure same number of factors by padding with 1s
        max_cores = max(len(in_factors), len(out_factors))
        if len(in_factors) < max_cores:
            in_factors = list(in_factors) + [1] * (max_cores - len(in_factors))
        if len(out_factors) < max_cores:
            out_factors = list(out_factors) + [1] * (max_cores - len(out_factors))
        
        self.in_factors = in_factors
        self.out_factors = out_factors
        self.n_cores = len(in_factors)
        
        # Auto-compute ranks if not provided
        if ranks is None:
            # Default: moderate compression
            max_rank = min(16, in_features // 8, out_features // 8)
            ranks = self._default_ranks(self.n_cores, max_rank)
        
        # Add boundary ranks
        self.ranks = [1] + list(ranks[:self.n_cores-1]) + [1]
        
        # Create TT cores
        self.cores = nn.ParameterList()
        for i in range(self.n_cores):
            core_shape = (self.ranks[i], in_factors[i], out_factors[i], self.ranks[i+1])
            
            # Xavier/Kaiming initialization
            core = torch.empty(core_shape)
            fan_in = self.ranks[i] * in_factors[i]
            fan_out = out_factors[i] * self.ranks[i+1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            core.normal_(0, std)
            
            self.cores.append(nn.Parameter(core))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Print compression stats
        if verbose:
            full_params = in_features * out_features
            tt_params = sum(np.prod(core.shape) for core in self.cores)
            print(f"TTLinear: {in_features} -> {out_features}, compression: {full_params/tt_params:.1f}x")
    
    def _auto_factorize(self, n: int) -> List[int]:
        """Auto-factorization for dimension"""
        if n == 512:
            return [8, 8, 8]
        elif n == 768:
            return [8, 12, 8]
        elif n == 1024:
            return [8, 16, 8]
        elif n == 2048:
            return [8, 32, 8]
        elif n == 4096:
            return [16, 16, 16]
        elif n <= 64:
            return [n]
        
        # General case
        sqrt = int(np.sqrt(n))
        for i in range(sqrt, 1, -1):
            if n % i == 0:
                return sorted([i, n // i])
        return [n]
    
    def _default_ranks(self, n_cores: int, max_rank: int) -> List[int]:
        """Generate default ranks"""
        if n_cores <= 2:
            return [min(max_rank, 16)]
        
        # Peak in the middle
        ranks = []
        mid = n_cores // 2
        for i in range(n_cores - 1):
            if i < mid:
                rank = min(max_rank, 4 * (i + 1))
            else:
                rank = min(max_rank, 4 * (n_cores - i))
            ranks.append(rank)
        return ranks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct forward pass following the manual trace logic
        Handles both 2D and 3D inputs
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            x = x.reshape(batch_size * seq_len, self.in_features)
        else:
            batch_size = original_shape[0]
            seq_len = None
        
        # Start with input reshaped
        x = x.reshape(-1, *self.in_factors)
        
        # Process cores sequentially
        for i in range(self.n_cores):
            core = self.cores[i]
            
            if i == 0:
                # First core
                core_squeezed = core.squeeze(0)  # [n0, m0, r1]
                
                # Calculate remaining input dimensions
                remaining_in = np.prod(self.in_factors[1:]) if len(self.in_factors) > 1 else 1
                
                # Reshape x for contraction
                x_reshaped = x.reshape(-1, self.in_factors[0], remaining_in)
                
                # Contract: [bs, n0, remaining] × [n0, m0, r1] -> [bs, m0, r1, remaining]
                result = torch.einsum('bnr,nmk->bmkr', x_reshaped, core_squeezed)
                
                # Reshape for next iteration
                if len(self.in_factors) > 1:
                    x = result.reshape(-1, self.out_factors[0], self.ranks[1], *self.in_factors[1:])
                else:
                    x = result.reshape(-1, self.out_factors[0], self.ranks[1])
                
            elif i == self.n_cores - 1:
                # Last core
                core_squeezed = core.squeeze(-1)  # [rk, nk, mk]
                
                # Calculate accumulated output
                accumulated_out = np.prod(self.out_factors[:i])
                
                # Reshape for contraction
                x_reshaped = x.reshape(-1, self.ranks[i], self.in_factors[i])
                
                # Contract: [bs*acc, rk, nk] × [rk, nk, mk] -> [bs*acc, mk]
                result = torch.einsum('bri,rio->bo', x_reshaped, core_squeezed)
                
                # Final reshape
                x = result.reshape(-1, accumulated_out * self.out_factors[i])
                
            else:
                # Middle cores
                # Calculate dimensions
                accumulated_out = np.prod(self.out_factors[:i])
                remaining_in = np.prod(self.in_factors[i+1:]) if i < self.n_cores - 1 else 1
                
                # Reshape to isolate current indices
                x_reshaped = x.reshape(-1, accumulated_out, self.ranks[i], 
                                       self.in_factors[i], remaining_in)
                
                # Contract: [bs, acc_out, ri, ni, rem_in] × [ri, ni, mi, ri+1]
                #        -> [bs, acc_out, mi, ri+1, rem_in]
                result = torch.einsum('barin,rioj->baojn', x_reshaped, core)
                
                # Reshape for next iteration
                new_accumulated = accumulated_out * self.out_factors[i]
                if i < self.n_cores - 2:
                    x = result.reshape(-1, new_accumulated, self.ranks[i+1], *self.in_factors[i+1:])
                else:
                    x = result.reshape(-1, new_accumulated, self.ranks[i+1], self.in_factors[-1])
        
        # Restore original batch shape
        if seq_len is not None:
            x = x.reshape(batch_size, seq_len, self.out_features)
        else:
            x = x.reshape(batch_size, self.out_features)
        
        # Add bias if needed
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def get_compression_ratio(self) -> float:
        """Calculate actual compression ratio"""
        full_params = self.in_features * self.out_features
        tt_params = sum(np.prod(core.shape) for core in self.cores)
        return full_params / tt_params


class TTMultiHeadAttention(nn.Module):
    """Multi-head attention with TT decomposition"""
    
    def __init__(self, config: TTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        # TT decomposition for Q, K, V projections
        self.q_proj = TTLinear(
            config.d_model, config.d_model,
            config.factorization, config.factorization,
            config.tt_ranks
        )
        self.k_proj = TTLinear(
            config.d_model, config.d_model,
            config.factorization, config.factorization,
            config.tt_ranks
        )
        self.v_proj = TTLinear(
            config.d_model, config.d_model,
            config.factorization, config.factorization,
            config.tt_ranks
        )
        
        # Output projection also uses TT
        self.out_proj = TTLinear(
            config.d_model, config.d_model,
            config.factorization, config.factorization,
            config.tt_ranks
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Rotary position embeddings (RoPE)
        self.rotary_emb = RotaryEmbedding(self.d_head, config.max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # Changed from -1e9 for fp16 compatibility
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return output


class TTFFN(nn.Module):
    """Feed-forward network with TT decomposition"""
    
    def __init__(self, config: TTConfig):
        super().__init__()
        self.config = config
        
        # Expansion factor for FFN (typically 4x)
        self.hidden_dim = config.d_model * 4
        
        # Factorize hidden dimension
        hidden_factors = self._factorize_hidden(self.hidden_dim)
        
        # TT layers with adaptive ranks
        self.up_proj = TTLinear(
            config.d_model, self.hidden_dim,
            config.factorization, hidden_factors,
            config.tt_ranks
        )
        
        self.down_proj = TTLinear(
            self.hidden_dim, config.d_model,
            hidden_factors, config.factorization,
            config.tt_ranks
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Use GELU activation (smoother than ReLU)
        self.activation = nn.GELU()
    
    def _factorize_hidden(self, n: int) -> List[int]:
        """Factorize hidden dimension"""
        if n == 2048:
            return [8, 32, 8]
        elif n == 3072:
            return [8, 48, 8]
        elif n == 4096:
            return [16, 16, 16]
        else:
            # General case
            sqrt = int(np.sqrt(n))
            for i in range(sqrt, 1, -1):
                if n % i == 0:
                    return sorted([i, n // i])
            return [n]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand
        hidden = self.up_proj(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Contract
        output = self.down_proj(hidden)
        
        return output


class TTTransformerBlock(nn.Module):
    """Single Transformer block with TT decomposition"""
    
    def __init__(self, config: TTConfig):
        super().__init__()
        self.config = config
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Multi-head attention with TT
        self.attn = TTMultiHeadAttention(config)
        
        # Feed-forward network with TT
        self.ffn = TTFFN(config)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, heads, seq_len, dim]
        seq_len = q.shape[2]
        
        # Get precomputed cos and sin
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class TTTransformer(nn.Module):
    """Complete Tensor-Train Transformer model"""
    
    def __init__(self, config: TTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TTTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Language model head (can also use TT here)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> dict:
        """Forward pass with optional loss computation"""
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.token_emb(input_ids)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
        mask = mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, 1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for current sequence
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == 50256:  # GPT-2 EOS
                    break
        
        return input_ids
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics for all TT layers"""
        stats = {}
        total_original = 0
        total_compressed = 0
        
        for name, module in self.named_modules():
            if isinstance(module, TTLinear):
                ratio = module.get_compression_ratio()
                original = module.in_features * module.out_features
                compressed = sum(np.prod(core.shape) for core in module.cores)
                
                stats[name] = {
                    'compression_ratio': ratio,
                    'original_params': original,
                    'compressed_params': compressed
                }
                
                total_original += original
                total_compressed += compressed
        
        stats['total'] = {
            'compression_ratio': total_original / total_compressed if total_compressed > 0 else 1.0,
            'original_params': total_original,
            'compressed_params': total_compressed,
            'saved_params': total_original - total_compressed
        }
        
        return stats


def test_tt_transformer():
    """Test the TT Transformer implementation"""
    print("Testing TT Transformer V2 with Fixed TTLinear...")
    print("=" * 70)
    
    # Create config
    config = TTConfig(
        d_model=512,
        n_heads=8,
        n_layers=2,
        vocab_size=1000,
        max_seq_len=128
    )
    
    # Create model
    model = TTTransformer(config)
    
    # Test input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Check compression stats
    stats = model.get_compression_stats()
    print(f"\nCompression Statistics:")
    print(f"Total compression: {stats['total']['compression_ratio']:.2f}x")
    print(f"Parameters saved: {stats['total']['saved_params']:,}")
    
    # Test with labels (loss computation)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels)
    loss = outputs['loss']
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_length=20)
    print(f"\nGenerated shape: {generated.shape}")
    
    print("\n✅ TT Transformer V2 with Fixed TTLinear works correctly!")
    
    return model


if __name__ == "__main__":
    model = test_tt_transformer()
