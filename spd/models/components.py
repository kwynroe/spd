import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.module_utils import init_param_


class Gate(nn.Module):
    """A gate that maps a single input to a single output."""

    def __init__(self, C: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((C,)))
        self.bias = nn.Parameter(torch.zeros((C,)))
        fan_val = 1  # Since each weight gets applied independently
        init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        return x * self.weight + self.bias


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, C: int, n_ci_mlp_neurons: int):
        super().__init__()
        self.n_ci_mlp_neurons = n_ci_mlp_neurons

        self.mlp_in = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
        self.in_bias = nn.Parameter(torch.zeros((C, n_ci_mlp_neurons)))
        self.mlp_out = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
        self.out_bias = nn.Parameter(torch.zeros((C,)))

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=n_ci_mlp_neurons, nonlinearity="linear")

    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        hidden = (
            einops.einsum(
                x,
                self.mlp_in,
                "... C, C n_ci_mlp_neurons -> ... C n_ci_mlp_neurons",
            )
            + self.in_bias
        )
        hidden = F.gelu(hidden)

        out = (
            einops.einsum(
                hidden,
                self.mlp_out,
                "... C n_ci_mlp_neurons, C n_ci_mlp_neurons -> ... C",
            )
            + self.out_bias
        )
        return out


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    NOTE: In the paper, we use V and U for A and B, respectively.

    The weight matrix W is decomposed as W = B^T @ A^T, where A and B are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, bias: Tensor | None):
        super().__init__()
        self.C = C

        self.A = nn.Parameter(torch.empty(d_in, C))
        self.B = nn.Parameter(torch.empty(C, d_out))
        self.bias = bias

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """B^T @ A^T"""
        return einops.einsum(self.A, self.B, "d_in C, C d_out -> d_out d_in")

    # @torch.compile
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.A, "... d_in, d_in C -> ... C")

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(component_acts, self.B, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
    ):
        super().__init__()
        self.C = C

        self.A = nn.Parameter(torch.empty(vocab_size, C))
        self.B = nn.Parameter(torch.empty(C, embedding_dim))

        init_param_(self.A, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """A @ B"""
        return einops.einsum(
            self.A, self.B, "vocab_size C, ... C embedding_dim -> vocab_size embedding_dim"
        )

    # @torch.compile
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.A[x]  # (batch pos C)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.B, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out


.

class AttentionComponent(nn.Module): 
    def __init__(
        self, 
        d_model: int,
        C: int,
        bias: Tensor | None = None,
        causal_mask: bool = True,
        attn_scores_normed: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.attn_scores_norm = self.d_model if attn_scores_normed else 1.0
        self.C = C
        self.use_causal_mask = causal_mask
        self.A = nn.Parameter(torch.empty(d_model, C))
        self.B = nn.Parameter(torch.empty(C, d_model))
        self.bias = bias
        self.is_attention_module = True

        init_param_(self.A, fan_val=d_model, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "... C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "d_model d_model"]:
        """B^T @ A^T"""
        return einops.einsum(self.A, self.B, "d_model C, C d_model -> d_model d_model")
    
    def causal_mask(self, scores: Float[Tensor, "batch seq1 seq2"]) -> Float[Tensor, "batch seq1 seq2"]:
        """Apply a causal mask to the attention scores."""
        batch, seq_len, _ = scores.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        return scores.masked_fill(causal_mask, float('-inf'))
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq seq"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor [batch, seq, d_model]
        Returns:
            Attention scores [batch, seq, seq]
        """
        if self.mask is not None:
            masked_A = self.A * self.mask.unsqueeze(0)
            masked_B = self.B * self.mask.unsqueeze(-1)
        else:
            masked_A = self.A
            masked_B = self.B
        
        # Construct QK^T from masked components
        QK_T = einops.einsum(masked_A, masked_B, "d_model C, C d_model -> d_model d_model")
        
        # Compute attention scores: x @ QK^T @ x^T
        scores = einops.einsum(x, QK_T, x, "batch seq1 d_model, d_model d_model, batch seq2 d_model -> batch seq1 seq2")/self.attn_scores_norm

        # Apply causal mask if needed
        if self.use_causal_mask:
            scores = self.causal_mask(scores)
        
        pattern = scores.softmax(dim=-1)
        return pattern
