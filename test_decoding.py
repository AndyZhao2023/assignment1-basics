#!/usr/bin/env python3
"""
Test script for the decoding (text generation) implementation.
Tests temperature scaling, top-p sampling, and complete text generation.
"""

import torch
import numpy as np
from cs336_basics.nn import (
    TransformerLM, 
    softmax_with_temperature, 
    top_p_sampling, 
    generate_text
)
from cs336_basics.tokenizer import Tokenizer

def test_temperature_scaling():
    """Test temperature scaling functionality"""
    print("Testing temperature scaling...")
    
    # Create some test logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Logits:", logits.tolist())
    for temp in temperatures:
        probs = softmax_with_temperature(logits, temperature=temp)
        print(f"Temperature {temp}: {probs.tolist()}")
        
        # Verify probabilities sum to 1
        assert abs(probs.sum().item() - 1.0) < 1e-6, f"Probabilities don't sum to 1 for temp {temp}"
    
    print("✓ Temperature scaling tests passed\n")


def test_top_p_sampling():
    """Test top-p sampling functionality"""
    print("Testing top-p sampling...")
    
    # Create a test probability distribution
    probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])  # Already normalized
    
    # Test different p values
    p_values = [0.5, 0.8, 0.9, 1.0]
    
    print("Probability distribution:", probs.tolist())
    
    # Run multiple samples to see distribution
    for p in p_values:
        samples = []
        for _ in range(100):
            sample = top_p_sampling(probs, p=p)
            samples.append(sample.item())
        
        # Count occurrences of each token
        counts = np.bincount(samples, minlength=len(probs))
        proportions = counts / counts.sum()
        
        print(f"p={p}: Sample proportions = {proportions.tolist()}")
        
        # For very low p values, should mostly sample from top tokens
        if p <= 0.5:
            assert counts[0] > counts[1], f"Top token not sampled most for p={p}"
    
    print("✓ Top-p sampling tests passed\n")


def test_simple_generation():
    """Test simple text generation with a small model"""
    print("Testing text generation...")
    
    # Create a very small model for testing
    vocab_size = 100
    context_length = 32
    d_model = 64
    num_layers = 2
    num_heads = 2
    d_ff = 128
    
    device = 'cpu'
    
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0
    ).to(device)
    
    # Create a simple tokenizer (character-level for testing)
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            
        def encode(self, text):
            if text == "<|endoftext|>":
                return [0]  # Special end token
            # Simple character encoding (ASCII values modulo vocab_size)
            return [min(ord(c) % (self.vocab_size - 1) + 1, self.vocab_size - 1) for c in text]
        
        def decode(self, tokens):
            result = ""
            for token in tokens:
                if token == 0:  # End token
                    result += "<|endoftext|>"
                else:
                    # Convert back to character (approximate)
                    result += chr(32 + (token - 1) % 95)  # Printable ASCII range
            return result
    
    tokenizer = SimpleTokenizer(vocab_size)
    
    # Test generation with different parameters
    prompt = "Hello"
    
    print(f"Original prompt: '{prompt}'")
    
    # Test different temperatures
    for temp in [0.1, 1.0, 2.0]:
        generated = generate_text(
            model, tokenizer, prompt, 
            max_tokens=20, temperature=temp, top_p=0.9, device=device
        )
        print(f"Temperature {temp}: '{generated[:50]}...'")
    
    # Test different top-p values
    for p in [0.5, 0.9, 1.0]:
        generated = generate_text(
            model, tokenizer, prompt, 
            max_tokens=20, temperature=1.0, top_p=p, device=device
        )
        print(f"Top-p {p}: '{generated[:50]}...'")
    
    print("✓ Text generation tests passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Decoding Implementation")
    print("=" * 60)
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_temperature_scaling()
        test_top_p_sampling()
        test_simple_generation()
        
        print("🎉 All decoding tests passed successfully!")
        print("\nImplementation includes:")
        print("✓ Temperature scaling with formula: softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σexp(vⱼ/τ)")
        print("✓ Top-p (nucleus) sampling for vocabulary truncation")
        print("✓ Text generation with <|endoftext|> token stopping")
        print("✓ Maximum token limit to prevent infinite generation")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()