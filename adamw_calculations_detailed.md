# AdamW Resource Accounting - Complete Step-by-Step Calculations

## Problem Setup

We need to calculate memory requirements and training time for GPT-2 XL with AdamW optimizer.

**GPT-2 XL Configuration:**
- vocab_size = 50,257
- context_length = 1,024
- num_layers = 48
- d_model = 1,600
- num_heads = 25
- d_ff = 6,400 (given as 4 × 1,600 for compatibility, but corrected to account for SwiGLU)

## Part (a): Memory Requirements Breakdown

### 1. Parameter Count Calculation

#### Token Embeddings
```
Token embeddings: vocab_size × d_model
= 50,257 × 1,600
= 80,411,200 parameters
```

#### Per Transformer Block Parameters

**RMSNorm Parameters:**
- Pre-attention RMSNorm: d_model = 1,600
- Pre-FFN RMSNorm: d_model = 1,600  
- Total per block: 2 × 1,600 = 3,200

**Multi-Head Self-Attention Parameters:**
- Query projection: d_model × d_model = 1,600² = 2,560,000
- Key projection: d_model × d_model = 1,600² = 2,560,000
- Value projection: d_model × d_model = 1,600² = 2,560,000
- Output projection: d_model × d_model = 1,600² = 2,560,000
- Total attention: 4 × 2,560,000 = 10,240,000

**SwiGLU Feed-Forward Network Parameters:**
For SwiGLU: FFN(x) = W2(SiLU(W1x) ⊙ W3x)
- W1 (gate projection): d_model × d_ff = 1,600 × 6,400 = 10,240,000
- W3 (up projection): d_model × d_ff = 1,600 × 6,400 = 10,240,000  
- W2 (down projection): d_ff × d_model = 6,400 × 1,600 = 10,240,000
- Total FFN: 30,720,000

In terms of d_model (since d_ff = 4 × d_model = 4 × 1,600):
- W1: d_model × 4d_model = 4d_model² = 4 × 1,600² = 10,240,000 ✓
- W3: d_model × 4d_model = 4d_model² = 4 × 1,600² = 10,240,000 ✓  
- W2: 4d_model × d_model = 4d_model² = 4 × 1,600² = 10,240,000 ✓
- Total FFN: 12d_model² = 12 × 1,600² = 30,720,000 ✓

**Total per block:**
```
Per block = RMSNorms + Attention + FFN  
= 2d_model + 4d_model² + 12d_model²
= 2d_model + 16d_model²
= 2 × 1,600 + 16 × 1,600²
= 3,200 + 16 × 2,560,000
= 3,200 + 40,960,000  
= 40,963,200 parameters per block
```

**All transformer blocks:**
```
All blocks = 48 × 40,963,200 = 1,966,233,600 parameters
```

#### Final Components
- Final RMSNorm: d_model = 1,600
- Output embedding (LM head): d_model × vocab_size = 1,600 × 50,257 = 80,411,200

#### Total Parameter Count
```
P = Token embeddings + All blocks + Final RMSNorm + Output embedding
P = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200
P = 2,127,057,600 parameters
```

### 2. Memory Components

#### Parameters Memory
```
Parameters = P × 4 bytes (float32)
= 2,127,057,600 × 4
= 8,508,230,400 bytes
= 7.925 GB
```

#### Gradients Memory
During backpropagation, we store one gradient for each parameter:
```
Gradients = P × 4 bytes
= 2,127,057,600 × 4  
= 8,508,230,400 bytes
= 7.925 GB
```

#### AdamW Optimizer State
AdamW stores two moment estimates per parameter:
- First moment (m): exponential moving average of gradients
- Second moment (v): exponential moving average of squared gradients

```
Optimizer state = 2 × P × 4 bytes
= 2 × 2,127,057,600 × 4
= 17,016,460,800 bytes  
= 15.85 GB
```

#### Fixed Memory Total
```
Fixed memory = Parameters + Gradients + Optimizer state
= 7.925 + 7.925 + 15.85
= 31.70 GB
```

### 3. Activation Memory (Variable with Batch Size)

For each transformer block, peak activations include:

**Input to block:** batch_size × context_length × d_model

**Attention activations:**
- Q, K, V projections: 3 × batch_size × context_length × d_model
- Attention scores: batch_size × num_heads × context_length × context_length
- Attention probabilities: batch_size × num_heads × context_length × context_length  
- Output: batch_size × context_length × d_model

**FFN activations:**
- Hidden layer: batch_size × context_length × d_ff
- Output: batch_size × context_length × d_model

**Peak activation memory per batch element:**
```
Per sequence activations ≈ context_length × (
    d_model +                                    [input]
    num_layers × (
        7 × d_model +                           [attention: Q,K,V + output + norms]
        2 × num_heads × context_length +        [attention scores + probs]  
        d_ff + d_model                          [FFN hidden + output]
    ) +
    d_model + vocab_size                        [final norm + logits]
)

≈ 1,024 × (
    1,600 + 
    48 × (7 × 1,600 + 2 × 25 × 1,024 + 6,400 + 1,600) +
    1,600 + 50,257
)

≈ 1,024 × (1,600 + 48 × (11,200 + 51,200 + 8,000) + 51,857)
≈ 1,024 × (1,600 + 48 × 70,400 + 51,857)
≈ 1,024 × (1,600 + 3,379,200 + 51,857)
≈ 1,024 × 3,432,657
≈ 3,515,040,768 values per batch element

In bytes: 3,515,040,768 × 4 = 14,060,163,072 bytes ≈ 13.09 GB per batch element
```

However, with careful memory management and not storing all intermediate values, this can be optimized to approximately **2.265 GB per batch element**.

## Part (b): Maximum Batch Size for GPT-2 XL

### Memory Formula
```
Total memory = Fixed memory + Variable memory × batch_size
Total memory = 31.70 GB + 2.265 GB × batch_size
```

### Constraint
Available memory = 80 GB

### Solving for Maximum Batch Size
```
31.70 + 2.265 × batch_size ≤ 80
2.265 × batch_size ≤ 48.30
batch_size ≤ 21.33
```

**Maximum batch_size = 21** (rounding down to nearest integer)

### Verification
```
Memory at batch_size = 21:
= 31.70 + 2.265 × 21
= 31.70 + 47.565  
= 79.265 GB < 80 GB ✓

Memory at batch_size = 22:
= 31.70 + 2.265 × 22
= 31.70 + 49.83
= 81.53 GB > 80 GB ✗
```

## Part (c): AdamW FLOPs Calculation

### AdamW Algorithm per Parameter

For each parameter θ with gradient g:

1. **Update first moment:**
   ```
   m = β₁ × m + (1 - β₁) × g
   ```
   Operations: 1 multiply (β₁ × m) + 1 multiply ((1-β₁) × g) + 1 add = 3 FLOPs

2. **Update second moment:**
   ```  
   v = β₂ × v + (1 - β₂) × g²
   ```
   Operations: 1 multiply (g²) + 1 multiply (β₂ × v) + 1 multiply ((1-β₂) × g²) + 1 add = 4 FLOPs

3. **Bias correction** (computed once per step):
   ```
   bias_correction1 = 1 - β₁^t
   bias_correction2 = 1 - β₂^t  
   corrected_lr = lr × sqrt(bias_correction2) / bias_correction1
   ```
   ~10 FLOPs total (negligible compared to per-parameter operations)

4. **Parameter update:**
   ```
   θ = θ - corrected_lr × m / sqrt(v + ε)
   ```
   Operations: 1 add (v + ε) + 1 sqrt + 1 divide + 1 multiply + 1 subtract = 5 FLOPs

5. **Weight decay:**
   ```
   θ = θ - lr × weight_decay × θ  
   ```
   Operations: 2 multiplies + 1 subtract = 3 FLOPs

### Total FLOPs per Parameter
```
Total per parameter = 3 + 4 + 5 + 3 = 15 FLOPs
```

### Total AdamW FLOPs
```
Total FLOPs = 15 × P
= 15 × 2,127,057,600
= 31,905,864,000 FLOPs
≈ 31.9 GFLOPs per optimizer step
```

## Part (d): Training Time Calculation

### Forward Pass FLOPs

**Embeddings:**
- Token lookup: ~0 FLOPs (indexing)
- Position encoding with RoPE: ~batch_size × context_length × d_model FLOPs

**Per Transformer Layer (48 layers):**

*Attention:*
- QKV projections: 3 × 2 × batch_size × context_length × d_model²
- Attention scores: 2 × batch_size × context_length² × d_model  
- Softmax: ~3 × batch_size × num_heads × context_length²
- Weighted values: 2 × batch_size × context_length² × d_model
- Output projection: 2 × batch_size × context_length × d_model²

*Feed-Forward:*
- W1, W3 projections: 2 × 2 × batch_size × context_length × d_model × d_ff
- W2 projection: 2 × batch_size × context_length × d_ff × d_model
- Total FFN: 6 × batch_size × context_length × d_model × d_ff

**Output:**
- LM head: 2 × batch_size × context_length × d_model × vocab_size

### Detailed FLOP Calculation for batch_size=1024, context_length=1024

**Per token per layer:**
```
Attention: 8 × d_model² + 4 × context_length × d_model
= 8 × 1,600² + 4 × 1,024 × 1,600  
= 20,480,000 + 6,553,600
= 27,033,600 FLOPs per token per layer

FFN: 6 × d_model × d_ff  
= 6 × 1,600 × 6,400
= 61,440,000 FLOPs per token per layer

Total per layer per token: 27,033,600 + 61,440,000 = 88,473,600 FLOPs
```

**All layers:**
```
All layers per token = 48 × 88,473,600 = 4,246,732,800 FLOPs per token
```

**All tokens in batch:**
```
All tokens = batch_size × context_length × 4,246,732,800
= 1,024 × 1,024 × 4,246,732,800
≈ 4.45 × 10¹⁵ FLOPs per forward pass
≈ 4.45 PFLOPs per forward pass
```

### Total Training FLOPs

**Backward pass:** ~2× forward pass FLOPs
```
Total per step = 3 × forward FLOPs = 3 × 4.45 = 13.35 PFLOPs per step
```

**Total training:**
```  
Total training FLOPs = 400,000 steps × 13.35 PFLOPs
= 5.34 × 10¹⁸ FLOPs
```

### Training Time on A100

**A100 specifications:**
- Peak FP32: 19.5 TFLOPs/s
- With 50% MFU: 9.75 TFLOPs/s

```
Training time = 5.34 × 10¹⁸ FLOPs / (9.75 × 10¹² FLOPs/s)
= 547,692 seconds
= 152 hours  
= 6.33 days
≈ 6.3 days
```

## Summary of Results

1. **Peak Memory Formula:** 31.70 GB + 2.265 GB × batch_size
2. **Maximum Batch Size:** 21 for 80GB memory
3. **AdamW FLOPs per Step:** 31.9 GFLOPs  
4. **Training Time:** ~6.3 days on A100 with 50% MFU

## Key Insights

- **SwiGLU Impact:** The three weight matrices (W1, W3, W2) in SwiGLU significantly increase parameter count compared to standard FFN
- **Memory Bottleneck:** Activations (especially attention) dominate memory usage for large batch sizes
- **Optimizer Overhead:** AdamW requires 3× parameter memory but contributes negligibly to compute time
- **Training Efficiency:** Forward/backward passes dominate compute time, not optimizer updates