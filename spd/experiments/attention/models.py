
from spd.models.base_components import BaseAttentionModule

from pathlib import Path
from typing import Any, Self

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run
import einops

from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class AttnModelPaths(BaseModel):
    """Paths to output files from a TMSModel training run."""

    attn_train_config: Path
    checkpoint: Path


class AttnModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vocab_size: PositiveInt
    d_model: PositiveInt
    seq_len: PositiveInt
    causal_mask: bool = True
    attn_scores_normed: bool = True
    
    
class SingleHeadAttentionModel(nn.Module):
    """Minimal model: embed → single attention → unembed."""
    
    def __init__(self, cfg: AttnModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.seq_len = cfg.seq_len
        self.vocab_size = cfg.vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Single attention layer (this will be targeted by SPD)
        self.attention = BaseAttentionModule(self.d_model)
        
        # Unembedding
        self.unembed = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward: embed → attention → unembed.
        
        Args:
            x: Token indices [batch, seq]
        Returns:
            Logits [batch, seq, vocab_size]
        """
        batch, seq = x.shape
        device = x.device
        
        # Embeddings
        token_embeds = self.token_embedding(x)  # [batch, seq, d_model]
        hidden = token_embeds  # [batch, seq, d_model]
        
        # Attention
        attn_patterns = self.attention(hidden)  # [batch, seq, seq]
        hidden = einops.einsum(attn_patterns, hidden, 
                             "batch seq1 seq2, batch seq2 d_model -> batch seq1 d_model")
        
        # Unembed
        logits = self.unembed(hidden)  # [batch, seq, vocab_size]
        
        return logits