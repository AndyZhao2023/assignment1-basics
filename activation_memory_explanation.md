# Understanding Activation Memory in Transformers

## What is Activation Memory?

**Activation memory** is the GPU memory required to store intermediate computational results (activations) from the forward pass that are needed later during backpropagation to compute gradients.

## Why Do We Need It?

During training, we do:
1. **Forward pass**: Input → computations → loss
2. **Backward pass**: Use chain rule to compute gradients, working backwards from loss

The backward pass needs intermediate values from the forward pass to compute gradients correctly.

## Simple Example

```python
# Forward pass
x = input                    # Need to store: x
h1 = Linear1(x)             # Need to store: h1  
h2 = ReLU(h1)               # Need to store: h2
output = Linear2(h2)        # Need to store: output
loss = CrossEntropy(output, targets)

# Backward pass  
# To compute grad_Linear2_weights: need h2 (input to Linear2)
# To compute grad_Linear1_weights: need x (input to Linear1) and h1 (output of Linear1)
```

**All of h1, h2, x, output must be stored in GPU memory between forward and backward pass.**

## Transformer Block Activation Memory

Let's trace through what needs to be stored for a batch of sequences:

### Input Shape
- **Input**: `[batch_size, sequence_length, d_model]`
- For GPT-2 XL: `[batch_size, 1024, 1600]`

### Step-by-Step Through One Transformer Block

#### 1. Input to Block
```python
# Shape: [batch_size, seq_len, d_model]
x_input = input  # STORE: needed for residual connection gradient
```

#### 2. First RMSNorm  
```python
# Shape: [batch_size, seq_len, d_model]  
x_norm1 = RMSNorm(x_input)  # STORE: needed for attention gradients
```

#### 3. Multi-Head Attention
```python
# QKV Projections - Shape: [batch_size, seq_len, d_model] each
Q = Linear_Q(x_norm1)  # STORE: needed for attention score gradients
K = Linear_K(x_norm1)  # STORE: needed for attention score gradients  
V = Linear_V(x_norm1)  # STORE: needed for weighted sum gradients

# Reshape for multi-head: [batch_size, num_heads, seq_len, d_k]
Q_heads = reshape(Q)  # STORE
K_heads = reshape(K)  # STORE
V_heads = reshape(V)  # STORE

# Attention scores: [batch_size, num_heads, seq_len, seq_len]
scores = Q_heads @ K_heads.transpose()  # STORE: needed for softmax gradients

# Attention probabilities: [batch_size, num_heads, seq_len, seq_len]  
attn_probs = softmax(scores)  # STORE: needed for weighted sum gradients

# Weighted values: [batch_size, num_heads, seq_len, d_k]
attn_output = attn_probs @ V_heads  # STORE

# Concatenate heads and project: [batch_size, seq_len, d_model]
attn_final = Linear_O(concat(attn_output))  # STORE
```

#### 4. First Residual Connection
```python
# Shape: [batch_size, seq_len, d_model]
x_residual1 = x_input + attn_final  # STORE: needed for second RMSNorm gradients
```

#### 5. Second RMSNorm
```python  
# Shape: [batch_size, seq_len, d_model]
x_norm2 = RMSNorm(x_residual1)  # STORE: needed for FFN gradients
```

#### 6. Feed-Forward Network (SwiGLU)
```python
# Shape: [batch_size, seq_len, d_ff] where d_ff = 4 * d_model
gate = Linear_W1(x_norm2)     # STORE: needed for SiLU gradients
up = Linear_W3(x_norm2)       # STORE: needed for element-wise multiply gradients  

# Shape: [batch_size, seq_len, d_ff]
gate_activated = SiLU(gate)   # STORE: needed for element-wise multiply gradients

# Shape: [batch_size, seq_len, d_ff]  
ffn_hidden = gate_activated * up  # STORE: needed for W2 gradients

# Shape: [batch_size, seq_len, d_model]
ffn_output = Linear_W2(ffn_hidden)  # STORE
```

#### 7. Second Residual Connection  
```python
# Shape: [batch_size, seq_len, d_model]
block_output = x_residual1 + ffn_output  # STORE: input to next block
```

### Memory Calculation Per Block

**Most Memory-Intensive Items:**
1. **Attention scores & probabilities**: `2 × [batch_size, num_heads, seq_len, seq_len]`
   - For GPT-2 XL: `2 × [batch_size, 25, 1024, 1024] = 2 × 25 × batch_size × 1,048,576`
   - This is **quadratic in sequence length** - the main memory bottleneck!

2. **Intermediate tensors**: Multiple `[batch_size, seq_len, d_model]` and `[batch_size, seq_len, d_ff]` tensors

### Total Activation Memory Formula

For **one transformer block**:
```
Memory ≈ batch_size × seq_len × 4 × [
    7 × d_model                           # Various d_model-sized tensors  
    + 2 × num_heads × seq_len            # Attention matrices (quadratic!)
    + d_ff                               # FFN hidden state
]
```

For **all 48 blocks in GPT-2 XL**:
```
Total ≈ batch_size × 48 × seq_len × 4 × [7 × 1600 + 2 × 25 × 1024 + 6400]
      ≈ batch_size × 48 × 1024 × 4 × [11,200 + 51,200 + 6,400]  
      ≈ batch_size × 48 × 1024 × 4 × 68,800
      ≈ batch_size × 13.7 GB
```

But with **memory optimizations** (gradient checkpointing, activation recomputation), this can be reduced to ~2.27 GB per batch element.

## Key Points

1. **Quadratic Growth**: Attention activations grow as O(seq_len²), making long sequences very expensive
2. **Batch Scaling**: Activation memory scales linearly with batch size  
3. **Not Shared**: Unlike parameters/gradients/optimizer state, activations are unique per batch
4. **Temporary**: Only needed between forward and backward pass, then can be freed
5. **Optimization Target**: Many techniques exist to reduce activation memory (gradient checkpointing, etc.)

## Why This Matters

For GPT-2 XL with batch_size=21:
- **Fixed memory** (params + gradients + optimizer): 31.70 GB
- **Activation memory**: 21 × 2.27 = 47.67 GB  
- **Total**: ~79.4 GB (fits in 80 GB!)

The activation memory often dominates for large batch sizes, which is why it's the limiting factor for how big a batch we can fit in GPU memory.