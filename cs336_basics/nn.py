import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
    
    SiLU(x) = x * sigmoid(x)
    
    Uses float32 for intermediate computations to ensure numerical stability.
    
    Args:
        x: Input tensor of arbitrary shape
        
    Returns:
        Tensor of the same shape with SiLU applied element-wise
    """
    # Store original dtype
    orig_dtype = x.dtype
    
    # Upcast to float32 for numerical stability
    x_float32 = x.to(torch.float32)
    
    # Compute sigmoid and multiply with input
    # SiLU(x) = x * sigmoid(x)
    output = x_float32 * torch.sigmoid(x_float32)
    
    # Cast back to original dtype
    return output.to(orig_dtype)


class Linear(nn.Module):
    """
    A linear transformation module that performs y = xW^T (no bias).
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Initialize weights according to section 3.4.1: N(0, 2/(d_in+d_out)) truncated at [-3σ, 3σ]
        std = (2 / (d_in + d_out)) ** 0.5
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Apply linear transformation to input.
        
        Args:
            x: Input tensor with shape (..., d_in)
            
        Returns:
            Output tensor with shape (..., d_out)
        """
        # Perform matrix multiplication: x @ weight.T
        # x shape: (..., d_in)
        # weight shape: (d_out, d_in)
        # weight.T shape: (d_in, d_out)
        # output shape: (..., d_out)
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    An embedding layer that maps token indices to d-dimensional vectors.
    
    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors (d_model)
        device: Device to store the parameters on
        dtype: Data type of the parameters
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # Initialize embeddings according to section 3.4.1: N(0, 1) truncated at [-3, 3]
        # Store with embedding_dim as the final dimension
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: Input tensor of token IDs with arbitrary shape
            
        Returns:
            Embeddings tensor with shape (..., embedding_dim)
        """
        # Perform embedding lookup without using nn.Embedding or nn.functional.embedding
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes the input by its root mean square and applies learned scaling.
    Uses float32 for internal computations to prevent overflow.
    
    Args:
        d_model: The dimension to normalize over (typically the model dimension)
        eps: Small constant for numerical stability (default: 1e-5)
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Apply RMS normalization to the input.
        
        Args:
            x: Input tensor with shape (..., d_model)
            
        Returns:
            Normalized tensor with the same shape as input
        """
        # Store original dtype
        orig_dtype = x.dtype
        
        # Upcast to float32 to prevent overflow when squaring
        x_float32 = x.to(torch.float32)
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Square the input
        x_squared = x_float32.pow(2)
        
        # Compute mean across the last dimension (d_model), keeping dims for broadcasting
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        
        # Add epsilon for numerical stability and take square root
        rms = torch.sqrt(mean_squared + self.eps)
        
        # Normalize by RMS
        x_normalized = x_float32 / rms
        
        # Cast back to original dtype
        x_normalized = x_normalized.to(orig_dtype)
        
        # Apply learned scaling weights
        return x_normalized * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network used in modern transformers like LLaMA.
    
    Implements: output = W2(SiLU(xW1^T) ⊙ xW3^T)
    
    Args:
        d_model: Input and output dimension
        d_ff: Hidden dimension (typically 8/3 * d_model, rounded to multiple of 64)
    """
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        
        # If d_ff not specified, compute it as 8/3 * d_model rounded to multiple of 64
        if d_ff is None:
            d_ff = int((8/3) * d_model)
            # Round up to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64
        
        # Three linear transformations (no bias, following modern conventions)
        self.w1 = Linear(d_model, d_ff)  # Gate projection
        self.w2 = Linear(d_ff, d_model)  # Down projection
        self.w3 = Linear(d_model, d_ff)  # Up projection
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Apply SwiGLU feed-forward network.
        
        Args:
            x: Input tensor with shape (..., d_model)
            
        Returns:
            Output tensor with same shape as input
        """
        # Store original dtype for final output
        orig_dtype = x.dtype
        
        # Gate path: apply W1 and SiLU activation
        gate = self.w1(x)
        gate = silu(gate)
        
        # Up projection path: apply W3
        up = self.w3(x)
        
        # Element-wise multiplication (gating)
        # Both gate and up have shape (..., d_ff)
        hidden = gate * up
        
        # Down projection: apply W2 to get back to d_model
        output = self.w2(hidden)
        
        return output