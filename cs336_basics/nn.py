import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable
import numpy.typing as npt
import os
from typing import BinaryIO, IO


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


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Compute softmax along a specified dimension with numerical stability.
    
    Uses the log-sum-exp trick to prevent numerical overflow:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to compute softmax
        
    Returns:
        Tensor of same shape with softmax applied along specified dimension
    """
    # Store original dtype
    orig_dtype = x.dtype
    
    # Upcast to float32 for numerical stability
    x_float32 = x.to(torch.float32)
    
    # Subtract the maximum value along the specified dimension for numerical stability
    # This prevents overflow in exp() while maintaining the same result
    # keepdim=True ensures the max has the same number of dimensions for broadcasting
    x_max = x_float32.max(dim=dim, keepdim=True).values
    x_shifted = x_float32 - x_max
    
    # Compute exponentials of shifted values (all <= 0, so exp <= 1)
    exp_x = torch.exp(x_shifted)
    
    # Sum exponentials along the specified dimension
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    # Normalize to get probabilities
    softmax_output = exp_x / sum_exp
    
    # Cast back to original dtype
    return softmax_output.to(orig_dtype)


def cross_entropy(inputs: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """
    Compute cross-entropy loss with numerical stability.
    
    Implements the formula: ℓi = − log softmax(oi)[xi+1]
    Then averages across all batch examples.
    
    Uses the log-sum-exp trick for numerical stability:
    log(softmax(x)) = x - log_sum_exp(x)
    
    Args:
        inputs: Unnormalized logits of shape (..., vocab_size) where ... represents 
                any number of batch dimensions
        targets: True class indices of shape (...) with values in [0, vocab_size-1]
        
    Returns:
        Scalar tensor with average cross-entropy loss across all examples
    """
    # Store original dtype
    orig_dtype = inputs.dtype
    
    # Upcast to float32 for numerical stability
    inputs_float32 = inputs.to(torch.float32)
    
    # Get shape information
    original_shape = inputs_float32.shape
    vocab_size = original_shape[-1]
    batch_dims = original_shape[:-1]
    
    # Calculate total number of examples across all batch dimensions
    total_batch_size = inputs_float32.numel() // vocab_size
    
    # Flatten all batch dimensions while keeping vocab_size as last dimension
    # (..., vocab_size) -> (total_batch_size, vocab_size)
    inputs_flat = inputs_float32.view(total_batch_size, vocab_size)
    
    # Flatten targets to match
    # (...) -> (total_batch_size,)
    targets_flat = targets.view(total_batch_size)
    
    # Extract logits for the correct classes using advanced indexing
    # inputs_flat[i, targets_flat[i]] gives the logit for the true class of example i
    correct_class_logits = inputs_flat[torch.arange(total_batch_size), targets_flat]
    
    # Compute log_sum_exp for each example (denominator of log softmax)
    # Subtract max for numerical stability (log-sum-exp trick)
    max_logits = inputs_flat.max(dim=1, keepdim=True).values
    shifted_logits = inputs_flat - max_logits
    log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=1)) + max_logits.squeeze(1)
    
    # Cross-entropy loss: ℓi = − log softmax(oi)[xi+1] = -(oi[xi+1] - log_sum_exp(oi))
    individual_losses = -(correct_class_logits - log_sum_exp)
    
    # Average across all examples
    avg_loss = individual_losses.mean()
    
    # Cast back to original dtype
    return avg_loss.to(orig_dtype)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"], 
    V: Float[Tensor, "... values d_v"],
    mask: Float[Tensor, "... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k + mask) V
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v)
        mask: Optional mask tensor of shape (..., queries, keys).
              Should contain 0 for valid positions and -inf for masked positions.
              
    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    # Get the dimension of keys for scaling
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores QK^T
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # scores: (..., queries, keys)
    scores = Q @ K.transpose(-2, -1)
    
    # Step 2: Scale by √d_k to prevent gradient vanishing
    scale = 1.0 / (d_k ** 0.5)
    scores = scores * scale
    
    # Step 3: Apply mask if provided
    if mask is not None:
        # Handle boolean mask: True means keep, False means mask out
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float('-inf'))
        else:
            # Assume mask contains 0 for valid positions and -inf for masked positions
            scores = scores + mask
    
    # Step 4: Apply softmax to get attention weights
    # softmax along the last dimension (keys)
    attention_weights = softmax(scores, dim=-1)
    
    # Step 5: Apply attention weights to values
    # attention_weights: (..., queries, keys), V: (..., keys, d_v)
    # output: (..., queries, d_v)
    output = attention_weights @ V
    
    return output


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


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in RoFormer.
    
    Applies rotation to pairs of dimensions based on position, encoding
    absolute position while maintaining relative position properties.
    
    Args:
        theta: Base for the geometric progression of frequencies (typically 10000)
        d_k: Dimension of query/key vectors (must be even)
        max_seq_len: Maximum sequence length to precompute
        device: Device to store precomputed buffers on
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Compute frequencies for each dimension pair
        # θ_k = Θ^(-2k/d) for k = 0, 1, ..., d/2-1
        # This matches θ_{i,k} = i × Θ^((2k-1)/d) when combined with positions
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = theta ** (-k / d_k)
        
        # Compute angles for all positions up to max_seq_len
        # angles[i, k] = i × θ_k
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_seq_len, d_k/2)
        
        # Precompute cos and sin values
        # Using register_buffer with persistent=False as recommended
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)
    
    def forward(self, x: Float[Tensor, "... seq_len d_k"], 
                token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)
            
        Returns:
            Tensor of same shape as x with RoPE applied
        """
        *batch_dims, seq_len, d_k = x.shape
        
        # Ensure we're working with float32 for numerical stability
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Get cos and sin values for the given positions
        # Handle arbitrary batch dimensions by flattening and reshaping
        flat_pos = token_positions.reshape(-1)
        cos = self.cos_cached[flat_pos].reshape(*token_positions.shape, self.d_k // 2)
        sin = self.sin_cached[flat_pos].reshape(*token_positions.shape, self.d_k // 2)
        
        # Reshape x to separate pairs of dimensions
        # (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
        x_reshape = x.reshape(*batch_dims, seq_len, self.d_k // 2, 2)
        
        # Extract even and odd indices (the pairs)
        x_even = x_reshape[..., 0]  # (..., seq_len, d_k/2)
        x_odd = x_reshape[..., 1]   # (..., seq_len, d_k/2)
        
        # Apply rotation matrix to each pair:
        # [cos  -sin] [x_even]   [x_even * cos - x_odd * sin]
        # [sin   cos] [x_odd ] = [x_even * sin + x_odd * cos]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Stack and reshape back to original shape
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.reshape(*batch_dims, seq_len, d_k)
        
        # Cast back to original dtype
        return x_rotated.to(orig_dtype)


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention mechanism.
    
    Implements the scaled dot-product attention across multiple heads:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Model dimension (input and output dimension)
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for causal mask
        rope: Optional RoPE module for positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.max_seq_len = max_seq_len
        self.rope = rope
        
        # Linear projections for Q, K, V, and output
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # Register causal mask buffer if max_seq_len is provided
        if max_seq_len is not None:
            causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            self.register_buffer('causal_mask', causal_mask, persistent=False)
        else:
            self.causal_mask = None
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"], 
                token_positions: Int[Tensor, "... seq_len"] = None) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply multi-head self-attention to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE
            
        Returns:
            Output tensor of same shape as input
        """
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
            
        # Linear projections
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)  # (..., seq_len, d_model)
        V = self.v_proj(x)  # (..., seq_len, d_model)
        
        # Reshape for multi-head attention
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        # Apply RoPE if provided
        if self.rope is not None and token_positions is not None:
            # Flatten batch and head dimensions for RoPE
            # (..., num_heads, seq_len, d_k) -> (batch*num_heads, seq_len, d_k)
            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)
            
            # Repeat token_positions for each head
            # (..., seq_len) -> (batch*num_heads, seq_len)
            pos_flat = token_positions.unsqueeze(-2).expand(*batch_dims, self.num_heads, seq_len).reshape(-1, seq_len)
            
            # Apply RoPE
            Q_flat = self.rope(Q_flat, pos_flat)
            K_flat = self.rope(K_flat, pos_flat)
            
            # Reshape back to multi-head format
            Q = Q_flat.reshape(*batch_dims, self.num_heads, seq_len, self.d_k)
            K = K_flat.reshape(*batch_dims, self.num_heads, seq_len, self.d_k)
        
        # Create causal mask
        # For causal attention, we want to mask out future positions (upper triangular)
        # True = keep, False = mask out
        if self.causal_mask is not None and seq_len <= self.max_seq_len:
            # self.causal_mask has True for positions to mask out, so invert it
            mask = ~self.causal_mask[:seq_len, :seq_len]
        else:
            # Create lower triangular mask (True = keep, False = mask out)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Apply scaled dot-product attention
        # Q, K, V: (..., num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (..., num_heads, seq_len, d_k)
        
        # Reshape back to original format
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k) -> (..., seq_len, d_model)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)
        
        # Final output projection
        output = self.output_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head self-attention and feed-forward network.
    
    Implements the pre-norm architecture:
    y = x + MultiHeadSelfAttention(RMSNorm(x))
    z = y + FFN(RMSNorm(y))
    
    Args:
        d_model: Model dimension (input and output dimension)
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length for causal mask and RoPE
        rope: Optional RoPE module for positional encoding
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = None, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Layer normalization modules
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope)
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"], 
                token_positions: Int[Tensor, "... seq_len"] = None) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply Transformer block to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE
            
        Returns:
            Output tensor of same shape as input
        """
        # First sub-layer: multi-head self-attention with residual connection
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_input = self.ln1(x)
        attn_output = self.attn(attn_input, token_positions)
        y = x + attn_output
        
        # Second sub-layer: feed-forward network with residual connection
        # z = y + FFN(RMSNorm(y))
        ffn_input = self.ln2(y)
        ffn_output = self.ffn(ffn_input)
        z = y + ffn_output
        
        return z


class TransformerLM(nn.Module):
    """
    Transformer Language Model that combines token embeddings, transformer blocks,
    and an output projection for next-token prediction.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        rope_theta: RoPE theta parameter
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Create shared RoPE instance
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, self.rope)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = RMSNorm(d_model)
        
        # Language model head (output projection)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """
        Forward pass of the Transformer Language Model.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, sequence_length)
            
        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Convert token IDs to embeddings
        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        
        # Generate token positions for RoPE
        token_positions = torch.arange(seq_len, device=token_ids.device).expand(batch_size, seq_len)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Apply final layer normalization
        x = self.ln_final(x)
        
        # Project to vocabulary size
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip the gradients of a set of parameters to have L2 norm at most max_l2_norm.
    
    Args:
        parameters: Collection of trainable parameters
        max_l2_norm: Maximum L2 norm for the combined gradients
        
    The gradients are modified in-place.
    """
    # Collect all gradients that exist (skip frozen parameters)
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    # If no gradients, nothing to clip
    if not gradients:
        return
    
    # Compute the total L2 norm of all gradients combined
    # Method 1: Compute individual norms, then combine (matches PyTorch's approach)
    total_norm = 0.0
    for grad in gradients:
        total_norm += grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    # Only clip if the norm exceeds the maximum
    if clip_coef < 1.0:
        for grad in gradients:
            grad.data.mul_(clip_coef)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and labels from a dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')
        
    Returns:
        Tuple of (inputs, labels) where:
        - inputs: shape (batch_size, context_length) 
        - labels: shape (batch_size, context_length), shifted by 1 from inputs
    """
    # Validate that dataset is long enough
    if len(dataset) < context_length + 1:
        raise ValueError(f"Dataset length {len(dataset)} is too short for context_length {context_length}")
    
    # Calculate valid range for starting indices
    max_start_idx = len(dataset) - context_length
    
    # Sample random starting indices
    start_indices = torch.randint(0, max_start_idx, (batch_size,))
    
    # Extract sequences for inputs and labels
    inputs = torch.zeros(batch_size, context_length, dtype=torch.long)
    labels = torch.zeros(batch_size, context_length, dtype=torch.long)
    
    for i, start_idx in enumerate(start_indices):
        # Input sequence: [start_idx, start_idx+1, ..., start_idx+context_length-1]
        inputs[i] = torch.from_numpy(dataset[start_idx:start_idx + context_length])
        # Label sequence: [start_idx+1, start_idx+2, ..., start_idx+context_length] 
        labels[i] = torch.from_numpy(dataset[start_idx + 1:start_idx + context_length + 1])
    
    # Move to specified device
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    return inputs, labels


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer, and iteration state to a checkpoint file.
    
    Args:
        model: PyTorch model to serialize
        optimizer: PyTorch optimizer to serialize
        iteration: Current training iteration number
        out: File path or file-like object to save checkpoint to
    """
    # Create checkpoint dictionary with all required state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Save using torch.save, which handles both file paths and file-like objects
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        src: File path or file-like object to load checkpoint from
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into
        
    Returns:
        The iteration number from the checkpoint
    """
    # Load checkpoint dictionary
    checkpoint = torch.load(src, map_location='cpu')
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the saved iteration number
    return checkpoint['iteration']


def softmax_with_temperature(logits: Float[Tensor, "... vocab_size"], 
                            temperature: float = 1.0, 
                            dim: int = -1) -> Float[Tensor, "... vocab_size"]:
    """
    Compute softmax with temperature scaling for text generation.
    
    Temperature scaling formula: softmax(v, τ)i = exp(vi/τ) / Σexp(vj/τ)
    
    Args:
        logits: Input logits of shape (..., vocab_size)
        temperature: Temperature parameter τ. Higher values make distribution more uniform,
                    lower values make it more peaked. Must be > 0.
        dim: Dimension along which to compute softmax
        
    Returns:
        Temperature-scaled probabilities of same shape as input
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Apply temperature scaling by dividing logits by temperature
    scaled_logits = logits / temperature
    
    # Apply standard softmax to scaled logits
    return softmax(scaled_logits, dim=dim)


def top_p_sampling(probs: Float[Tensor, "vocab_size"], p: float = 0.9) -> Int[Tensor, ""]:
    """
    Sample from top-p (nucleus) sampling distribution.
    
    Top-p sampling keeps only the smallest set of tokens whose cumulative 
    probability mass is at least p, then samples from this truncated distribution.
    
    Args:
        probs: Probability distribution over vocabulary of shape (vocab_size,)
        p: Cumulative probability threshold. Must be in (0, 1]
        
    Returns:
        Sampled token index
    """
    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")
    
    # Sort probabilities in descending order and get sorted indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Find where cumulative probability exceeds p
    # Keep tokens up to and including the first one that makes cumsum >= p
    keep_mask = cumulative_probs <= p
    
    # Ensure we keep at least one token (the most probable one)
    if not keep_mask.any():
        keep_mask[0] = True
    else:
        # Include one more token after the threshold to ensure we have something to sample from
        last_keep_idx = keep_mask.nonzero()[-1].item()
        if last_keep_idx < len(keep_mask) - 1:
            keep_mask[last_keep_idx + 1] = True
    
    # Zero out probabilities of tokens we don't want to keep
    filtered_probs = sorted_probs.clone()
    filtered_probs[~keep_mask] = 0.0
    
    # Renormalize the remaining probabilities
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    # Sample from the filtered distribution
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1).item()
    
    # Convert back to original vocabulary index
    return sorted_indices[sampled_idx]


def generate_text(model: 'TransformerLM', 
                  tokenizer,
                  prompt: str,
                  max_tokens: int = 100,
                  temperature: float = 1.0,
                  top_p: float = 0.9,
                  device: str = 'cpu') -> str:
    """
    Generate text using a transformer language model with temperature scaling and top-p sampling.
    
    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt string
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for scaling logits (higher = more random)
        top_p: Nucleus sampling parameter (keep tokens with cumulative prob <= top_p)
        device: Device to run generation on
        
    Returns:
        Generated text string including the original prompt
    """
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Get end-of-text token ID
    endoftext_token = None
    try:
        if hasattr(tokenizer, 'encode'):
            endoftext_ids = tokenizer.encode("<|endoftext|>")
            if endoftext_ids:
                endoftext_token = endoftext_ids[0]
    except:
        # If encoding fails, continue without end-of-text detection
        pass
    
    generated_tokens = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions for current sequence
            # Only use the last context_length tokens if sequence is too long
            current_length = input_tensor.shape[1]
            if current_length > model.context_length:
                input_tensor = input_tensor[:, -model.context_length:]
            
            logits = model(input_tensor)  # Shape: (1, seq_len, vocab_size)
            
            # Get logits for the last token (next token prediction)
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            # Apply temperature scaling
            probs = softmax_with_temperature(next_token_logits, temperature=temperature)
            
            # Sample using top-p sampling
            next_token = top_p_sampling(probs, p=top_p)
            
            # Add the new token to our sequence
            generated_tokens.append(next_token.item())
            
            # Check if we generated the end-of-text token
            if endoftext_token is not None and next_token.item() == endoftext_token:
                break
            
            # Update input tensor for next iteration
            next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
    
    # Decode the generated tokens back to text
    return tokenizer.decode(generated_tokens)
