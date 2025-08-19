# AdamW Resource Accounting Solution

## Part (a): Peak Memory Requirements

### Model Parameters
Total parameters consist of:
- **Token Embeddings**: `vocab_size × d_model`
- **Transformer Blocks** (per layer):
  - RMSNorm gain: `2 × d_model` (pre-attention + pre-FFN)
  - Multi-head attention:
    - Q, K, V projections: `3 × d_model × d_model`
    - Output projection: `d_model × d_model`
  - Feed-forward network with SwiGLU (with d_ff = 4 × d_model):
    - W1 (gate): `d_model × d_ff = d_model × 4d_model`
    - W3 (up): `d_model × d_ff = d_model × 4d_model`
    - W2 (down): `d_ff × d_model = 4d_model × d_model`
- **Final RMSNorm**: `d_model`
- **Output Embedding**: `d_model × vocab_size`

**Total Parameters**:
```
P = vocab_size × d_model                                    [token embeddings]
  + num_layers × (2 × d_model                               [RMSNorms in block]
                  + 4 × d_model²                             [attention QKVO]
                  + 12 × d_model²)                           [FFN with SwiGLU: W1, W3, W2]
  + d_model                                                  [final RMSNorm]
  + d_model × vocab_size                                    [output embedding]

P = 2 × vocab_size × d_model + num_layers × (2 × d_model + 16 × d_model²) + d_model
```

### Gradients
Gradients require the same memory as parameters:
```
G = P
```

### Optimizer State (AdamW)
AdamW maintains two moment estimates (m and v) for each parameter:
```
O = 2 × P
```

### Activations
Peak activations during forward pass (assuming we keep all for backward):

Per sequence position, we need:
- **Input/Residual stream**: `batch_size × context_length × d_model`
- **Per Transformer block**:
  - After first RMSNorm: `batch_size × context_length × d_model`
  - Q, K, V projections: `3 × batch_size × context_length × d_model`
  - Attention scores (Q^T K): `batch_size × num_heads × context_length × context_length`
  - Attention probs (after softmax): `batch_size × num_heads × context_length × context_length`
  - Attention output: `batch_size × context_length × d_model`
  - After second RMSNorm: `batch_size × context_length × d_model`
  - FFN hidden: `batch_size × context_length × d_ff = batch_size × context_length × 4d_model`
  - FFN output: `batch_size × context_length × d_model`
- **Final RMSNorm output**: `batch_size × context_length × d_model`
- **Logits**: `batch_size × context_length × vocab_size`

**Peak Activations** (keeping all intermediate values):
```
A = batch_size × context_length × d_model                    [input]
  + num_layers × (
      batch_size × context_length × d_model                  [post-RMSNorm1]
      + 3 × batch_size × context_length × d_model            [Q,K,V]
      + 2 × batch_size × num_heads × context_length²         [attention scores + probs]
      + batch_size × context_length × d_model                [attention output]
      + batch_size × context_length × d_model                [post-RMSNorm2]
      + batch_size × context_length × 4d_model               [FFN hidden]
      + batch_size × context_length × d_model                [FFN output]
    )
  + batch_size × context_length × d_model                    [final RMSNorm]
  + batch_size × context_length × vocab_size                 [logits]

A = batch_size × context_length × (
    d_model + vocab_size + num_layers × (
      7d_model + 4d_model + 2num_heads × context_length/d_model
    ) + d_model
  )
```

### Total Memory (in float32, 4 bytes per value)
```
Memory = 4 × (P + G + O + A)
       = 4 × (P + P + 2P + A)
       = 4 × (4P + A)
```

## Part (b): GPT-2 XL Maximum Batch Size

### GPT-2 XL Configuration
- vocab_size = 50,257
- context_length = 1,024
- num_layers = 48
- d_model = 1,600
- num_heads = 25
- d_ff = 6,400

### Parameter Count
```
P = 2 × 50,257 × 1,600 + 48 × (2 × 1,600 + 16 × 1,600²) + 1,600
  = 160,822,400 + 48 × (3,200 + 40,960,000) + 1,600
  = 160,822,400 + 48 × 40,963,200 + 1,600
  = 160,822,400 + 1,966,233,600 + 1,600
  = 2,127,057,600 parameters
```

### Memory Formula
With 4P for parameters+gradients+optimizer state:
```
Memory_fixed = 4 × 4 × 2,127,057,600 bytes = 34,032,921,600 bytes ≈ 31.70 GB

Memory_variable = 4 × batch_size × 1,024 × (
    1,600 + 50,257 + 48 × (11,200 + 2 × 25 × 1,024/1,600) + 1,600
  )
= 4 × batch_size × 1,024 × (54,457 + 48 × (11,200 + 32))
= 4 × batch_size × 1,024 × (54,457 + 539,136)
= 4 × batch_size × 1,024 × 593,593 bytes
= 2,432,188,416 × batch_size bytes
≈ 2.265 × batch_size GB
```

### Maximum Batch Size for 80GB
```
Total_memory = 31.70 + 2.265 × batch_size ≤ 80
2.265 × batch_size ≤ 48.30
batch_size ≤ 21.33
```

**Maximum batch_size = 21**

**Expression**: Memory = 31.70 GB + 2.265 × batch_size GB

## Part (c): FLOPs for One AdamW Step

For each parameter, AdamW performs:
1. **Update first moment**: `m = β₁m + (1-β₁)g`
   - 2 multiplications + 1 addition per parameter
2. **Update second moment**: `v = β₂v + (1-β₂)g²`
   - 3 multiplications + 1 addition per parameter
3. **Bias correction and update**: `θ = θ - α_t × m/√(v+ε)`
   - Bias correction scalars (computed once): ~10 FLOPs
   - Per parameter: 1 sqrt + 1 division + 2 multiplications + 2 additions
4. **Weight decay**: `θ = θ - αλθ`
   - 2 multiplications + 1 subtraction per parameter

**Total per parameter**: ~13 operations

**Total FLOPs for AdamW step**:
```
FLOPs_AdamW = 13 × P = 13 × 2,127,057,600 ≈ 27.65 GFLOPs
```

## Part (d): Training Time on A100

### Forward Pass FLOPs
From transformer accounting analysis:
- Embeddings: `2 × batch_size × context_length × vocab_size × d_model`
- Attention (per layer): `4 × batch_size × context_length² × d_model² + 4 × batch_size × context_length × d_model²`
- FFN (per layer): `2 × batch_size × context_length × d_model × d_ff × 2`
- Output: `2 × batch_size × context_length × d_model × vocab_size`

For batch_size=1024, context_length=1024:
```
FLOPs_forward ≈ 330 TFLOPs per step
```

### Total Training FLOPs
With backward = 2 × forward:
```
FLOPs_per_step = 3 × FLOPs_forward = 990 TFLOPs
Total_FLOPs = 400,000 × 990 TFLOPs = 3.96 × 10²⁰ FLOPs
```

### Time on A100
- A100 peak: 19.5 TFLOPs/s for float32
- With 50% MFU: 9.75 TFLOPs/s effective
```
Time = 3.96 × 10²⁰ / (9.75 × 10¹² × 86400) seconds
     = 3.96 × 10²⁰ / 8.424 × 10¹⁷ seconds
     ≈ 470 seconds per step × 400,000 steps
     ≈ 5.4 days
```

**Training would take approximately 5.4 days**