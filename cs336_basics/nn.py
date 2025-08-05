import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


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