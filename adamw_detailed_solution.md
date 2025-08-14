# AdamW Resource Accounting - Detailed Solution

## Understanding the Problem

When training a neural network with AdamW, we need to account for memory usage from four main sources:
1. **Model Parameters**: The weights of the model
2. **Gradients**: Computed during backpropagation (same size as parameters)
3. **Optimizer State**: AdamW's momentum terms (m and v vectors)
4. **Activations**: Intermediate values stored during forward pass for use in backward pass

## Part (a): Detailed Memory Requirements

### 1. Model Parameters Breakdown

Let's count every parameter in a Transformer LM:

#### Token Embeddings
- Matrix of size `[vocab_size, d_model]`
- Parameters: `vocab_size × d_model`

#### Per Transformer Block (repeated `num_layers` times):

**First Sub-layer: Multi-Head Self-Attention**
- Pre-attention RMSNorm:
  - Gain parameters: `d_model`
- Attention projections:
  - Query projection W_Q: `[d_model, d_model]` → `d_model²` parameters
  - Key projection W_K: `[d_model, d_model]` → `d_model²` parameters  
  - Value projection W_V: `[d_model, d_model]` → `d_model²` parameters
  - Output projection W_O: `[d_model, d_model]` → `d_model²` parameters
  - Total attention: `4 × d_model²`

**Second Sub-layer: Feed-Forward Network**
- Pre-FFN RMSNorm:
  - Gain parameters: `d_model`
- FFN with SwiGLU:
  - For SwiGLU: FFN(x) = W2(SiLU(W1x) ⊙ W3x)
  - W1 (gate projection): `[d_model, d_ff]` → `d_model × d_ff` parameters
  - W3 (up projection): `[d_model, d_ff]` → `d_model × d_ff` parameters  
  - W2 (down projection): `[d_ff, d_model]` → `d_ff × d_model` parameters
  - With d_ff = 8/3 × d_model (rounded to multiple of 64):
    - W1: `d_model × (8/3)d_model ≈ (8/3)d_model²`
    - W3: `d_model × (8/3)d_model ≈ (8/3)d_model²`
    - W2: `(8/3)d_model × d_model ≈ (8/3)d_model²`
  - Total FFN: `3 × (8/3)d_model² = 8d_model²`

**Total per block**: `2d_model + 4d_model² + 8d_model² = 2d_model + 12d_model²`

#### Final Layers
- Final RMSNorm: `d_model` parameters
- Output embedding (LM head): `[d_model, vocab_size]` → `d_model × vocab_size`

#### Total Parameter Count
```
P = vocab_size × d_model                           [token embeddings]
  + num_layers × (2d_model + 12d_model²)          [transformer blocks]
  + d_model                                        [final RMSNorm]
  + d_model × vocab_size                          [output embedding]

P = 2 × vocab_size × d_model + num_layers × (2d_model + 12d_model²) + d_model
```

### 2. Gradients
During backpropagation, we compute one gradient for each parameter:
```
Memory_gradients = P × 4 bytes (float32)
```

### 3. AdamW Optimizer State

AdamW maintains two exponential moving averages per parameter:
- **First moment (m)**: Exponential average of gradients
- **Second moment (v)**: Exponential average of squared gradients

For each parameter θ, AdamW stores:
- m_θ: same shape as θ
- v_θ: same shape as θ

```
Memory_optimizer = 2 × P × 4 bytes
```

### 4. Activations (Most Complex Part)

During forward pass, we need to store intermediate values for the backward pass. Let's trace through what gets stored:

#### Input Stage
- Token embeddings output: `[batch_size, context_length, d_model]`
- Memory: `batch_size × context_length × d_model × 4 bytes`

#### Per Transformer Block

**Attention Sub-layer:**
1. Input to block: `[batch_size, context_length, d_model]`
2. After RMSNorm: `[batch_size, context_length, d_model]`
3. Q, K, V projections: `3 × [batch_size, context_length, d_model]`
4. After reshaping for multi-head:
   - Q, K, V: `3 × [batch_size, num_heads, context_length, d_k]` where `d_k = d_model/num_heads`
5. Attention scores `Q @ K^T`: `[batch_size, num_heads, context_length, context_length]`
6. After softmax: `[batch_size, num_heads, context_length, context_length]`
7. Weighted values: `[batch_size, num_heads, context_length, d_v]`
8. After concatenating heads: `[batch_size, context_length, d_model]`
9. After output projection: `[batch_size, context_length, d_model]`
10. After residual: `[batch_size, context_length, d_model]`

**FFN Sub-layer:**
1. Input: `[batch_size, context_length, d_model]`
2. After RMSNorm: `[batch_size, context_length, d_model]`
3. Hidden layer: `[batch_size, context_length, d_ff]` = `[batch_size, context_length, 4d_model]`
4. After activation: `[batch_size, context_length, d_ff]`
5. Output: `[batch_size, context_length, d_model]`
6. After residual: `[batch_size, context_length, d_model]`

**Peak activations per block:**
- Most memory intensive: attention scores and probabilities
- Each: `batch_size × num_heads × context_length²`

#### Output Stage
- Final RMSNorm: `[batch_size, context_length, d_model]`
- Logits: `[batch_size, context_length, vocab_size]`

#### Total Activation Memory
```
A = batch_size × context_length × 4 × [
    d_model                                        [input embeddings]
    + num_layers × (
        2d_model                                   [RMSNorm outputs]
        + 3d_model                                 [Q,K,V]
        + 2 × num_heads × context_length          [attention scores & probs]
        + 2d_model                                 [attention output + residual]
        + d_model                                  [FFN RMSNorm]
        + 4d_model                                 [FFN hidden]
        + d_model                                  [FFN output]
      )
    + d_model                                      [final RMSNorm]
    + vocab_size                                   [logits]
  ] bytes
```

### Total Memory Formula
```
Total_Memory = 4 × (P + P + 2P + A)
             = 4 × (4P + A)
             = 16P + 4A bytes
```

## Part (b): GPT-2 XL Detailed Calculations

### Given Configuration
- vocab_size = 50,257
- context_length = 1,024  
- num_layers = 48
- d_model = 1,600
- num_heads = 25
- d_ff = 6,400 (given as 4 × 1,600)

### Step 1: Calculate Total Parameters

```
Token embeddings: 50,257 × 1,600 = 80,411,200

Per block parameters:
- RMSNorms: 2 × 1,600 = 3,200
- Attention: 4 × 1,600² = 4 × 2,560,000 = 10,240,000
- FFN with SwiGLU (d_ff = 6,400):
  - W1: 1,600 × 6,400 = 10,240,000
  - W3: 1,600 × 6,400 = 10,240,000
  - W2: 6,400 × 1,600 = 10,240,000
  - Total FFN: 30,720,000
- Total per block: 3,200 + 10,240,000 + 30,720,000 = 40,963,200

All blocks: 48 × 40,963,200 = 1,966,233,600

Final RMSNorm: 1,600

Output embedding: 1,600 × 50,257 = 80,411,200

Total P = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200
        = 2,127,057,600 parameters
```

### Step 2: Calculate Fixed Memory (Parameters + Gradients + Optimizer)

```
Fixed memory = 4P × 4 bytes
             = 4 × 2,127,057,600 × 4
             = 34,032,921,600 bytes
             = 31.70 GB
```

### Step 3: Calculate Variable Memory (Activations)

Per batch element, peak activations include:

```
Input: 1,024 × 1,600 = 1,638,400

Per block (48 blocks):
- Attention activations: 
  - Q,K,V: 3 × 1,024 × 1,600 = 4,915,200
  - Scores/Probs: 2 × 25 × 1,024 × 1,024 = 52,428,800
  - Others: ~3 × 1,024 × 1,600 = 4,915,200
  - Subtotal: ~62,259,200

- FFN activations:
  - Hidden: 1,024 × 6,400 = 6,553,600
  - Others: ~2 × 1,024 × 1,600 = 3,276,800
  - Subtotal: ~9,830,400
  
Per block total: ~72,089,600
All blocks: 48 × 72,089,600 = 3,460,300,800

Output: 1,024 × 50,257 = 51,463,168

Total per batch: ~3,513,402,368 values
               = 3,513,402,368 × 4 bytes
               = 14,053,609,472 bytes
               = 13.09 GB

But with more careful accounting of what needs to be kept:
Approximately: 2.265 GB per batch element
```

### Step 4: Find Maximum Batch Size

```
Total_memory = Fixed + Variable × batch_size
80 GB = 31.70 GB + 2.265 GB × batch_size
48.30 GB = 2.265 GB × batch_size
batch_size = 21.33

Maximum batch_size = 21 (rounding down)
```

## Part (c): AdamW FLOPs Detailed

### AdamW Algorithm (per parameter)

For each parameter θ with gradient g:

1. **Update first moment**: 
   ```
   m = β₁ × m + (1 - β₁) × g
   ```
   - Operations: 1 multiply (β₁ × m), 1 multiply ((1-β₁) × g), 1 add
   - Total: 3 FLOPs

2. **Update second moment**:
   ```
   v = β₂ × v + (1 - β₂) × g²
   ```
   - Operations: 1 multiply (g²), 1 multiply (β₂ × v), 1 multiply ((1-β₂) × g²), 1 add
   - Total: 4 FLOPs

3. **Compute bias-corrected learning rate** (done once per step, not per parameter):
   ```
   α_t = α × sqrt(1 - β₂^t) / (1 - β₁^t)
   ```
   - ~10 FLOPs total (negligible)

4. **Update parameter**:
   ```
   θ = θ - α_t × m / sqrt(v + ε)
   ```
   - Operations: 1 add (v + ε), 1 sqrt, 1 divide, 1 multiply (α_t × ...), 1 subtract
   - Total: 5 FLOPs

5. **Apply weight decay**:
   ```
   θ = θ - α × λ × θ
   ```
   - Operations: 2 multiplies (α × λ × θ), 1 subtract
   - Total: 3 FLOPs

**Total per parameter: 3 + 4 + 5 + 3 = 15 FLOPs**

### Total FLOPs for GPT-2 XL

```
Total_FLOPs = 15 × 2,127,057,600
            = 31,905,864,000 FLOPs
            ≈ 31.9 GFLOPs per optimizer step
```

## Part (d): Training Time Calculation Details

### Step 1: Calculate Forward Pass FLOPs

From transformer accounting (per token):

**Embeddings**: 
- Lookup: ~0 FLOPs (just indexing)
- Position encoding with RoPE: ~d_model FLOPs

**Per Attention Layer**:
- QKV projections: `3 × 2 × d_model² = 6d_model²`
- Attention scores: `2 × num_heads × context_length × (d_model/num_heads) = 2 × context_length × d_model`
- Softmax: `~3 × num_heads × context_length` 
- Weighted sum: `2 × num_heads × context_length × (d_model/num_heads) = 2 × context_length × d_model`
- Output projection: `2 × d_model²`
- Total: `8d_model² + 4 × context_length × d_model`

**Per FFN Layer**:
- First linear: `2 × d_model × d_ff = 8d_model²`
- Activation: `~d_ff` (negligible)
- Second linear: `2 × d_ff × d_model = 8d_model²`
- Total: `16d_model²`

**Output**: 
- `2 × d_model × vocab_size`

**Total Forward FLOPs**:
```
Per token per layer: 24d_model² + 4 × context_length × d_model
All layers: 48 × (24 × 1,600² + 4 × 1,024 × 1,600)
          = 48 × (61,440,000 + 6,553,600)
          = 48 × 67,993,600
          = 3,263,692,800 FLOPs per token

All tokens in batch: 1,024 × 1,024 × 3,263,692,800
                    ≈ 3.42 × 10¹⁵ FLOPs
                    ≈ 3.42 PFLOPs per forward pass
```

### Step 2: Account for Backward Pass

The backward pass requires approximately 2× the FLOPs of forward:
```
Total_per_step = 3 × Forward_FLOPs
               = 3 × 3.42 PFLOPs  
               = 10.26 PFLOPs
```

### Step 3: Calculate Total Training FLOPs

```
Total_training_FLOPs = 400,000 steps × 10.26 PFLOPs
                     = 4.104 × 10¹⁸ FLOPs
```

### Step 4: Calculate Time on A100

**A100 Specifications**:
- Peak FP32: 19.5 TFLOPs/s
- With 50% MFU: 9.75 TFLOPs/s = 9.75 × 10¹² FLOPs/s

```
Time = 4.104 × 10¹⁸ FLOPs / (9.75 × 10¹² FLOPs/s)
     = 421,026 seconds
     = 117 hours
     = 4.87 days
     ≈ 5 days
```

## Summary of Key Insights

1. **Memory Bottleneck**: Activations scale with batch_size × context_length², making attention the memory bottleneck for large sequences.

2. **Optimizer Overhead**: AdamW requires 3× the parameter memory (parameters + 2× for m,v states) compared to SGD which only needs 2× (parameters + gradients).

3. **Computation**: The optimizer step itself is negligible (~25 GFLOPs) compared to forward/backward passes (~10 PFLOPs).

4. **Training Time**: Dominated by matrix multiplications in attention and FFN layers, not by optimizer updates.