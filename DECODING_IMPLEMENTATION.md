# CS336 Assignment 1: Decoding (3 Points) Implementation

This document describes the implementation of the "Decoding (3 points)" problem from Section 6 "Generating text" of the CS336 assignment.

## Overview

The decoding implementation provides text generation capabilities for transformer language models using temperature scaling and top-p (nucleus) sampling. This allows for controllable text generation with proper handling of vocabulary truncation and stopping conditions.

## Implementation Details

### 1. Temperature Scaling

**File**: `cs336_basics/nn.py` - `softmax_with_temperature()`

Implements the mathematical formula specified in the assignment:
```
softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σexp(vⱼ/τ)
```

**Key Features:**
- Divides logits by temperature parameter τ before applying softmax
- Lower temperature (τ < 1) makes distribution more peaked/deterministic  
- Higher temperature (τ > 1) makes distribution more uniform/random
- Maintains numerical stability using existing `softmax()` function
- Validates temperature parameter (must be > 0)

**Usage:**
```python
from cs336_basics.nn import softmax_with_temperature

logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
probs = softmax_with_temperature(logits, temperature=2.0)
```

### 2. Top-p (Nucleus) Sampling

**File**: `cs336_basics/nn.py` - `top_p_sampling()`

Implements nucleus sampling algorithm that keeps only the smallest set of tokens whose cumulative probability mass is at least p.

**Algorithm:**
1. Sort probabilities in descending order
2. Compute cumulative probabilities 
3. Find cutoff where cumulative probability ≥ p
4. Zero out probabilities below cutoff
5. Renormalize remaining probabilities
6. Sample from truncated distribution

**Key Features:**
- Dynamically adjusts vocabulary size based on probability distribution
- Always keeps at least one token (most probable)
- Handles edge cases properly (p ≤ 0 or p > 1)
- Uses multinomial sampling for final token selection

**Usage:**
```python
from cs336_basics.nn import top_p_sampling

probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])  # Must sum to 1
token = top_p_sampling(probs, p=0.9)
```

### 3. Text Generation

**File**: `cs336_basics/nn.py` - `generate_text()`

Complete autoregressive text generation function that combines all components.

**Key Features:**
- Uses temperature scaling for probability adjustment
- Applies top-p sampling for token selection
- Handles context window limits (truncates if sequence > model.context_length)
- Stops generation when `<|endoftext|>` token is produced
- Respects maximum token limit to prevent infinite generation
- Robust error handling for tokenizer compatibility

**Usage:**
```python
from cs336_basics.nn import generate_text

generated_text = generate_text(
    model=trained_model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_tokens=50,
    temperature=1.0,
    top_p=0.9,
    device='cuda'
)
```

## Test Adapters

**File**: `tests/adapters.py`

Added three adapter functions for testing integration:

- `run_generate_text()` - Main text generation adapter
- `run_softmax_with_temperature()` - Temperature scaling adapter  
- `run_top_p_sampling()` - Top-p sampling adapter

These follow the existing adapter pattern and can be used by the test suite.

## Demonstration Scripts

### Basic Testing: `test_decoding.py`
- Tests temperature scaling with various τ values
- Tests top-p sampling with different p thresholds
- Tests complete text generation pipeline
- Verifies mathematical correctness

### Comprehensive Demo: `demo_decoding.py`
- Shows practical effects of different temperature values
- Demonstrates vocabulary truncation with top-p sampling
- Compares generation outputs across parameter settings
- Provides educational examples with clear explanations

## Usage Examples

### Character-Level Generation
```python
# Simple character-level tokenizer
class CharTokenizer:
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, tokens): return ''.join(chr(t) for t in tokens)

tokenizer = CharTokenizer()
generated = generate_text(model, tokenizer, "Hello", max_tokens=100)
```

### BPE Tokenizer Generation
```python
from cs336_basics.tokenizer import Tokenizer

# Use trained BPE tokenizer
tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
generated = generate_text(
    model=trained_model,
    tokenizer=tokenizer,
    prompt="The quick brown fox",
    max_tokens=200,
    temperature=0.8,
    top_p=0.95
)
```

## Key Implementation Features

### Numerical Stability
- All computations use existing stable implementations (`softmax()`)
- Proper handling of edge cases and invalid parameters
- Robust error handling for tokenizer compatibility

### Performance Considerations  
- Efficient tensor operations using PyTorch
- Minimal memory overhead
- Context window management for long sequences

### Compatibility
- Works with any PyTorch model that outputs logits
- Compatible with any tokenizer that implements `encode()/decode()`
- Flexible parameter validation and defaults

## Testing

Run the demonstration:
```bash
uv run python demo_decoding.py
```

Run basic tests:
```bash
uv run python test_decoding.py
```

The implementation passes all tests and demonstrates proper functionality across different parameter settings.

## Conclusion

This implementation successfully addresses all requirements of the "Decoding (3 points)" problem:

✅ **Temperature scaling**: Implements exact mathematical formula  
✅ **Top-p sampling**: Proper nucleus sampling algorithm  
✅ **Text generation**: Complete autoregressive generation  
✅ **Stopping conditions**: Handles `<|endoftext|>` token and max_tokens  
✅ **Integration**: Works with existing model and tokenizer components  
✅ **Testing**: Comprehensive test coverage and demonstrations

The implementation is ready for use with trained models on TinyStories and OpenWebText datasets as specified in the assignment.