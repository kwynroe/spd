import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

import torch
import einops
from torch import nn, Tensor
from jaxtyping import Float


class BaseAttentionModule(nn.Module):
    """Simple attention module that can be replaced by AttentionComponent.
    
    This module computes attention patterns using a learnable QK^T matrix.
    It's designed to have the same interface as AttentionComponent for easy replacement.
    """
    
    def __init__(self, d_model: int, causal_mask: bool = True, attn_scores_normed: bool = True):
        super().__init__()
        self.d_model = d_model
        self.causal_mask = causal_mask
        self.attn_scores_normed = attn_scores_normed
        self.is_attention_module = True  # Flag for SPD component detection
        
        # Learnable QK^T matrix - this will be decomposed by SPD
        self.qk_weights = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # Normalization factor
        self.attn_scores_norm = d_model**0.5 if attn_scores_normed else 1.0
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq seq"]:
        """Forward pass computing attention patterns.
        
        Args:
            x: Input tensor [batch, seq, d_model]
        Returns:
            Attention patterns [batch, seq, seq]
        """
        # Compute attention scores: x @ QK^T @ x^T
        scores = einops.einsum(x, self.qk_weights, x, 
                             "batch seq1 d_model, d_model d_model, batch seq2 d_model -> batch seq1 seq2")
        
        # Apply normalization
        scores = scores / self.attn_scores_norm
        
        # Apply causal mask if enabled
        if self.causal_mask:
            seq_len = x.shape[1]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax to get attention patterns
        patterns = scores.softmax(dim=-1)
        
        return patterns
