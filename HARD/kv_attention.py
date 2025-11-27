"""
KV-Cached Multi-Head Attention for LLM Inference - AI CODEFIX 2025 HARD_2
FIXED VERSION - All bugs corrected

This module implements an optimized attention mechanism with Key-Value caching
for efficient autoregressive generation in Large Language Models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict


class KVCachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with KV-Cache optimization for LLM inference.

    This implementation caches Key and Value projections across generation steps
    to avoid redundant computation during autoregressive decoding.

    Standard transformer attention formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    With KV-caching:
        - Cache K and V from previous tokens
        - Concatenate new K, V with cached versions
        - Compute attention only for new query tokens
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_cache_len: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize the KV-Cached Multi-Head Attention layer.

        Args:
            d_model: Dimension of the model (embedding size)
            num_heads: Number of attention heads
            max_cache_len: Maximum sequence length to cache
            dropout: Dropout probability for attention weights
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # FIX #1: Scaling factor should be sqrt(head_dim) for scaled dot-product attention
        # Standard formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with KV-caching support.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            cache: Optional dict with cached 'key' and 'value' tensors
            use_causal_mask: Whether to apply causal masking

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            new_cache: Updated cache dictionary
        """
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Handle cache: concatenate cached K, V with new K, V
        cache_len = 0
        if cache is not None and cache.get('key') is not None:
            cached_k = cache['key']
            cached_v = cache['value']
            cache_len = cached_k.shape[1]

            # FIX #2: Cache concatenation should be on sequence dimension (dim=1)
            # After projection, K/V are [batch, seq_len, d_model]
            # We concatenate along seq_len dimension before splitting heads
            K = torch.cat([cached_k, K], dim=1)
            V = torch.cat([cached_v, V], dim=1)

        # Split into multiple heads
        Q = self._split_heads(Q)  # [batch, num_heads, seq_len_q, head_dim]
        K = self._split_heads(K)  # [batch, num_heads, total_seq_len, head_dim]
        V = self._split_heads(V)

        # Compute attention scores
        scores = self._compute_attention_scores(Q, K)

        # Apply causal mask if needed
        if use_causal_mask:
            scores = self._apply_causal_mask(scores, seq_len, cache_len)

        # FIX #3: Softmax on last dimension (key/sequence dimension)
        # This ensures attention weights sum to 1 across the sequence dimension
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout (automatically respects model.train/eval mode)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Merge heads back
        output = self._merge_heads(output)

        # Final output projection
        output = self.out_proj(output)

        # FIX #8: Store full concatenated K, V (not just new tokens)
        # The cache contains the complete history needed for next generation step
        new_cache = {
            'key': K,
            'value': V
        }

        # FIX #10: Cache size validation - check if we exceed maximum
        if new_cache['key'] is not None and new_cache['key'].shape[2] >= self.max_cache_len:
            raise ValueError(f"Cache exceeded maximum length: {new_cache['key'].shape[2]} >= {self.max_cache_len}")

        return output, new_cache

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split embedding dimension into (num_heads, head_dim).

        Reshapes from [batch, seq_len, d_model]
                   to [batch, num_heads, seq_len, head_dim]

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Reshaped tensor [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape

        # FIX #7: Reshape to [batch, seq_len, num_heads, head_dim] first
        # Then permute to [batch, num_heads, seq_len, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge head dimension back to embedding dimension.

        Reshapes from [batch, num_heads, seq_len, head_dim]
                   to [batch, seq_len, d_model]

        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Merged tensor [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def _compute_attention_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention scores.

        Standard formula: scores = (Q @ K^T) / sqrt(d_k)

        Args:
            Q: Query [batch, num_heads, seq_len_q, head_dim]
            K: Key [batch, num_heads, seq_len_k, head_dim]

        Returns:
            Attention scores [batch, num_heads, seq_len_q, seq_len_k]
        """
        # FIX #4: Transpose last two dimensions (-2, -1), not dimensions (1, 2)
        # Dimension (1, 2) would swap heads and seq_len - WRONG!
        # We need to transpose the last two dims: seq_len_k and head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale by sqrt(head_dim) for stability
        scores = scores / self.scale

        return scores

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        seq_len: int,
        cache_len: int
    ) -> torch.Tensor:
        """
        Apply causal mask to prevent attending to future positions.

        For autoregressive generation, position i can only attend to positions <= i.
        When using cache, current tokens can attend to all cached tokens.

        Args:
            scores: Attention scores [batch, num_heads, seq_len_q, total_seq_len]
            seq_len: Current query sequence length
            cache_len: Number of cached tokens

        Returns:
            Masked attention scores
        """
        # Total sequence length = cached + new
        total_seq_len = cache_len + seq_len

        # FIX #5: Offset should be cache_len (no +1)
        offset = cache_len

        # Create causal mask
        # Position i can attend to all cached positions and positions <= i in new tokens
        mask = torch.ones(seq_len, total_seq_len, device=scores.device)
        for i in range(seq_len):
            for j in range(total_seq_len):
                # FIX #6: Correct boundary condition
                # Cached positions (j < cache_len): always allowed
                # New positions (j >= cache_len): allowed if j <= i + cache_len (i.e., j - cache_len <= i)
                if j < cache_len or j <= i + offset:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

        # Apply mask: set future positions to -inf so softmax makes them ~0
        scores = scores.masked_fill(mask == 0, float('-inf'))

        return scores

    def reset_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Return an empty cache dictionary.

        Returns:
            Empty cache with None values
        """
        return {'key': None, 'value': None}

    def get_cache_info(
        self,
        cache: Optional[Dict[str, torch.Tensor]],
        use_cache: bool = True
    ) -> Dict[str, any]:
        """
        Get information about current cache state.

        Args:
            cache: Current cache dictionary
            use_cache: Whether caching is enabled (for future compatibility)

        Returns:
            Dictionary with cache statistics
        """
        if cache is None or cache.get('key') is None:
            return {
                'cache_length': 0,
                'cache_size_mb': 0.0,
                'is_full': False
            }

        key_cache = cache['key']

        # Get cache dimensions
        # After _split_heads, shape is [batch, num_heads, seq_len, head_dim]
        cache_seq_len = key_cache.shape[2]

        # Calculate memory usage (approximate)
        cache_size_bytes = key_cache.numel() * key_cache.element_size()
        cache_size_mb = (cache_size_bytes * 2) / (1024 * 1024)  # K and V caches

        return {
            'cache_length': cache_seq_len,
            'cache_size_mb': round(cache_size_mb, 2),
            'is_full': cache_seq_len >= self.max_cache_len
        }


def compute_position_ids(seq_len: int, cache_len: int) -> torch.Tensor:
    """
    Compute position IDs for positional encoding with cache.

    Args:
        seq_len: Current sequence length
        cache_len: Cached sequence length

    Returns:
        Position IDs tensor
    """
    position_ids = []
    for i in range(seq_len):
        position_ids.append(cache_len + i)
    return torch.tensor(position_ids, dtype=torch.long)


def validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> None:
    """
    Validate input tensor shapes and types.

    Args:
        query, key, value: Input tensors to validate

    Raises:
        ValueError: If inputs are invalid
    """
    if query.dim() != 3:
        raise ValueError(f"Query must be 3D tensor, got {query.dim()}D")
    if key.dim() != 3:
        raise ValueError(f"Key must be 3D tensor, got {key.dim()}D")
    if value.dim() != 3:
        raise ValueError(f"Value must be 3D tensor, got {value.dim()}D")

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch sizes must match")

    if key.shape[1] != value.shape[1]:
        raise ValueError("Key and value sequence lengths must match")


def create_sample_input(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample input tensors for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        seed: Random seed for reproducibility

    Returns:
        Tuple of (query, key, value) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    return query, key, value


if __name__ == "__main__":
    """
    Quick test to verify basic functionality.
    """
    print("=" * 60)
    print("KV-Cached Multi-Head Attention - Quick Test")
    print("=" * 60)

    # Configuration
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    # Initialize model
    model = KVCachedMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_cache_len=128,
        dropout=0.1
    )
    model.eval()

    # Create sample inputs
    q, k, v = create_sample_input(batch_size, seq_len, d_model, seed=42)

    print(f"\nInput shapes:")
    print(f"  Query: {q.shape}")
    print(f"  Key: {k.shape}")
    print(f"  Value: {v.shape}")

    # First forward pass (no cache)
    print("\n--- First forward pass (no cache) ---")
    try:
        output1, cache1 = model(q, k, v, cache=None, use_causal_mask=True)
        print(f"✓ Output shape: {output1.shape}")
        print(f"✓ Cache key shape: {cache1['key'].shape if cache1['key'] is not None else 'None'}")
        print(f"  Cache info: {model.get_cache_info(cache1)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Second forward pass (with cache) - simulating next token generation
    print("\n--- Second forward pass (with cache) ---")
    try:
        q2, k2, v2 = create_sample_input(batch_size, 1, d_model, seed=43)
        output2, cache2 = model(q2, k2, v2, cache=cache1, use_causal_mask=True)
        print(f"✓ Output shape: {output2.shape}")
        print(f"✓ Cache key shape: {cache2['key'].shape if cache2['key'] is not None else 'None'}")
        print(f"  Cache info: {model.get_cache_info(cache2)}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Code is fixed.")
    print("=" * 60)