
from spd.models.base_components import BaseAttentionModule

from pathlib import Path
from typing import Any
from typing_extensions import Self

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
    """Paths to output files from a AttnModel training run."""

    attn_train_config: Path
    checkpoint: Path


class AttnModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vocab_size: PositiveInt
    d_model: PositiveInt
    seq_len: PositiveInt
    causal_mask: bool = True
    attn_scores_normed: bool = True
    n_trigrams: NonNegativeInt = 32
    
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
        attn_out = self.attention(hidden)  # [batch, seq, d_model]
        # Unembed
        hidden = attn_out + hidden  # Residual connection
        logits = self.unembed(hidden)  # [batch, seq, vocab_size]
        
        return logits
    
    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["AttnModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained AttnModel
            attn_model_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = AttnModelPaths(
                attn_train_config=Path(path).parent / "attn_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.attn_train_config) as f:
            attn_train_config_dict = yaml.safe_load(f)

        attn_config = AttnModelConfig(**attn_train_config_dict["attn_model_config"])
        attn = cls(config=attn_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        attn.load_state_dict(params)

        if attn_config.tied_weights:
            attn.tie_weights_()

        return attn, attn_train_config_dict