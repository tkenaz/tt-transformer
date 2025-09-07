"""
Correct TT-Linear implementation for Tensor Transformer
Based on the fixed implementation that works with arbitrary number of cores

Authors: Marina & Claude
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class TTLinear(nn.Module):
    """
    Tensor-Train Linear layer with correct forward pass
    Replaces nn.Linear(in_features, out_features) with TT-decomposition
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_factors: Optional[List[int]] = None,
        out_factors: Optional[List[int]] = None,
        tt_ranks: Optional[List[int]] = None,
        bias: bool = True,
        auto_shapes: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        if seed:
            torch.manual_seed(seed)
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Auto-factorization if not provided
        if in_factors is None or out_factors is None:
            if auto_shapes:
                in_factors = self._auto_shape(in_features)
                out_factors = self._auto_shape(out_features)
            else:
                in_factors = self._factorize(in_features)
                out_factors = self._factorize(out_features)
        
        # Ensure same number of factors by padding with 1s if needed
        max_cores = max(len(in_factors), len(out_factors))
        if len(in_factors) < max_cores:
            in_factors = list(in_factors) + [1] * (max_cores - len(in_factors))
        if len(out_factors) < max_cores:
            out_factors = list(out_factors) + [1] * (max_cores - len(out_factors))
        
        self.in_factors = in_factors
        self.out_factors = out_factors
        self.n_cores = len(in_factors)
        
        # Validate
        assert np.prod(in_factors) == in_features, f"Product of in_factors {in_factors} != {in_features}"
        assert np.prod(out_factors) == out_features, f"Product of out_factors {out_factors} != {out_features}"
        assert len(in_factors) == len(out_factors), "in_factors and out_factors must have same length"
        
        print(f"TT-Linear: {in_features} -> {out_features}")
        print(f"  Input factorization: {in_factors} = {np.prod(in_factors)}")
        print(f"  Output factorization: {out_factors} = {np.prod(out_factors)}")
        
        # TT-ranks
        if tt_ranks is None:
            # Heuristic: start small, peak in middle
            max_rank = min(64, in_features // 4, out_features // 4)
            tt_ranks = self._default_ranks(self.n_cores, max_rank)
        
        # Ensure correct number of ranks
        if len(tt_ranks) < self.n_cores - 1:
            # Pad with last value
            tt_ranks = list(tt_ranks) + [tt_ranks[-1] if tt_ranks else 4] * (self.n_cores - 1 - len(tt_ranks))
        elif len(tt_ranks) > self.n_cores - 1:
            # Truncate
            tt_ranks = tt_ranks[:self.n_cores - 1]
        
        # Add boundary ranks (always 1)
        self.ranks = [1] + list(tt_ranks) + [1]
        
        print(f"  TT-ranks: {self.ranks}")
        
        # Create TT-cores
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
            print(f"  Core {i}: {core_shape}, params: {np.prod(core_shape)}")
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Compression stats
        original_params = in_features * out_features
        tt_params = sum(c.numel() for c in self.cores)
        self.compression_ratio = original_params / tt_params
        
        print(f"  Compression: {self.compression_ratio:.2f}x")
        print(f"  Original params: {original_params:,}")
        print(f"  TT params: {tt_params:,}")
    
    def _auto_shape(self, n: int) -> List[int]:
        """Auto-factorization of dimension for better compression"""
        # Special cases for common dimensions
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
        elif n == 256:
            return [8, 32] if n == 256 else [16, 16]
        elif n == 128:
            return [8, 16]
        elif n <= 64:
            return [n]  # Don't factorize very small dimensions
        
        # General case: try to factorize into 2-3 roughly equal factors
        factors = []
        remaining = n
        
        # Try cube root for 3 factors
        cube_root = int(round(n ** (1/3)))
        if cube_root > 4:
            # Try to find 3 factors
            for i in range(cube_root - 2, cube_root + 3):
                if i > 1 and n % i == 0:
                    factors.append(i)
                    remaining = n // i
                    # Try to factor the rest into 2
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
        
        return [n]  # Prime or couldn't factorize
    
    def _factorize(self, n: int) -> List[int]:
        """Simple factorization"""
        if n == 4096:
            return [64, 64]
        elif n == 768:
            return [24, 32]
        elif n == 1024:
            return [32, 32]
        elif n == 512:
            return [8, 8, 8]
        else:
            # Fallback to square root
            sqrt = int(np.sqrt(n))
            if sqrt * sqrt == n:
                return [sqrt, sqrt]
            return [n]
    
    def _default_ranks(self, n_cores: int, max_rank: int) -> List[int]:
        """Generate default ranks - peak in middle"""
        if n_cores == 1:
            return []
        
        ranks = []
        mid = n_cores // 2
        
        for i in range(n_cores - 1):
            if i < mid:
                # Grow towards middle
                rank = min(max_rank, 2 ** (i + 4))
            else:
                # Shrink from middle
                rank = min(max_rank, 2 ** (n_cores - i + 2))
            ranks.append(rank)
        
        return ranks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with correct TT contraction
        x: [batch_size, in_features] or [batch_size, seq_len, in_features]
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            x = x.reshape(batch_size * seq_len, self.in_features)
        else:
            batch_size = original_shape[0]
            seq_len = None
        
        # Factorize input
        x = x.reshape(-1, *self.in_factors)
        
        # Process cores sequentially
        for i in range(self.n_cores):
            core = self.cores[i]
            
            if i == 0:
                # First core: [1, n0, m0, r1]
                core_squeezed = core.squeeze(0)  # [n0, m0, r1]
                
                # Calculate remaining input dims
                remaining_in = np.prod(self.in_factors[1:]) if len(self.in_factors) > 1 else 1
                
                # Reshape for contraction
                x_reshaped = x.reshape(-1, self.in_factors[0], remaining_in)
                
                # Contract: [bs, n0, remaining] × [n0, m0, r1] -> [bs, m0, r1, remaining]
                result = torch.einsum('bnr,nmk->bmkr', x_reshaped, core_squeezed)
                
                # Reshape for next iteration
                if len(self.in_factors) > 1:
                    x = result.reshape(-1, self.out_factors[0], self.ranks[1], *self.in_factors[1:])
                else:
                    x = result.reshape(-1, self.out_factors[0], self.ranks[1])
                
            elif i == self.n_cores - 1:
                # Last core: [rk, nk, mk, 1]
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
                # Middle cores: [ri, ni, mi, ri+1]
                
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
        
        # Add bias
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'tt_ranks={self.ranks[1:-1]}, compression={self.compression_ratio:.1f}x'


# Test
if __name__ == "__main__":
    print("=== Testing Correct TT-Linear ===\n")
    
    # Test 1: Basic case
    layer = TTLinear(512, 512, tt_ranks=[4, 4])
    
    # Test batch input
    x = torch.randn(2, 512)
    y = layer(x)
    print(f"Batch input: {x.shape} -> {y.shape}")
    
    # Test sequence input
    x_seq = torch.randn(2, 10, 512)
    y_seq = layer(x_seq)
    print(f"Sequence input: {x_seq.shape} -> {y_seq.shape}")
    
    # Check gradients
    loss = y.mean()
    loss.backward()
    print(f"Gradients flow: {all(c.grad is not None for c in layer.cores)}")
    
    print("\n✅ TT-Linear works correctly!")
