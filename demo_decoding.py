#!/usr/bin/env python3
"""
Demonstration of the Decoding (3 points) implementation.
Shows temperature scaling, top-p sampling, and text generation functionality.
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


def demonstrate_temperature_effects():
    """Demonstrate how temperature affects probability distributions"""
    print("=" * 60)
    print("Temperature Scaling Demonstration")
    print("=" * 60)
    
    # Create realistic logits (like from a language model)
    vocab = ["the", "cat", "dog", "ran", "jumped", "quickly", "slowly", ".", ",", "!"]
    logits = torch.tensor([3.2, 2.8, 1.5, 2.1, 1.8, 0.9, 0.6, 2.5, 1.2, 0.4])
    
    print("Vocabulary:", vocab)
    print("Raw logits:", [f"{x:.2f}" for x in logits.tolist()])
    print()
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for temp in temperatures:
        probs = softmax_with_temperature(logits, temperature=temp)
        
        # Show top 3 most probable tokens
        top_probs, top_indices = torch.topk(probs, 3)
        
        print(f"Temperature {temp}:")
        print(f"  Formula: exp(logit/{temp}) / Σexp(logits/{temp})")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            word = vocab[idx]
            print(f"  #{i+1}: '{word}' -> {prob:.4f}")
        print()


def demonstrate_top_p_effects():
    """Demonstrate how top-p sampling works"""
    print("=" * 60)
    print("Top-p (Nucleus) Sampling Demonstration")
    print("=" * 60)
    
    # Create a probability distribution
    vocab = ["the", "cat", "dog", "ran", "jumped", "quickly", "slowly", ".", ",", "!"]
    probs = torch.tensor([0.35, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])
    
    print("Vocabulary:", vocab)
    print("Probabilities:", [f"{x:.3f}" for x in probs.tolist()])
    print()
    
    # Calculate cumulative probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    
    print("Sorted by probability (cumulative):")
    for i, (prob, idx, cum) in enumerate(zip(sorted_probs, sorted_indices, cumsum)):
        word = vocab[idx]
        print(f"  {i+1}. '{word}': {prob:.3f} (cumulative: {cum:.3f})")
    print()
    
    p_values = [0.5, 0.7, 0.9, 1.0]
    
    for p in p_values:
        print(f"Top-p sampling with p={p}:")
        
        # Show which tokens would be kept
        keep_indices = []
        cumulative = 0.0
        for i, (prob, idx) in enumerate(zip(sorted_probs, sorted_indices)):
            if cumulative + prob <= p or i == 0:  # Always keep at least one
                keep_indices.append(idx.item())
                cumulative += prob
            else:
                if cumulative < p:  # Add one more to reach threshold
                    keep_indices.append(idx.item())
                break
        
        kept_words = [vocab[i] for i in keep_indices]
        print(f"  Keeps {len(kept_words)} tokens: {kept_words}")
        
        # Sample a few times to show variety
        samples = []
        for _ in range(10):
            sample_idx = top_p_sampling(probs, p=p)
            samples.append(vocab[sample_idx.item()])
        
        from collections import Counter
        sample_counts = Counter(samples)
        print(f"  10 samples: {dict(sample_counts)}")
        print()


def demonstrate_text_generation():
    """Demonstrate complete text generation pipeline"""
    print("=" * 60)
    print("Text Generation Demonstration")
    print("=" * 60)
    
    # Initialize a small model for demonstration
    vocab_size = 256  # Character-level
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=64,
        d_model=128,
        num_layers=4,
        num_heads=8,
        d_ff=256,
        rope_theta=10000.0
    )
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Simple character-level tokenizer for demo
    class CharTokenizer:
        def __init__(self):
            self.char_to_id = {chr(i): i for i in range(256)}
            self.id_to_char = {i: chr(i) for i in range(256)}
            # Set a special end token
            self.char_to_id['<|endoftext|>'] = 255
            self.id_to_char[255] = '<|endoftext|>'
        
        def encode(self, text):
            if text == "<|endoftext|>":
                return [255]
            return [min(ord(c), 254) for c in text]
        
        def decode(self, tokens):
            result = ""
            for token in tokens:
                if token == 255:
                    result += "<|endoftext|>"
                elif 0 <= token < 256:
                    try:
                        result += chr(token)
                    except:
                        result += "?"
            return result
    
    tokenizer = CharTokenizer()
    
    prompts = ["Hello", "The cat", "Once upon"]
    
    print("Comparing different generation settings:")
    print()
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        
        # Test different temperature values
        for temp in [0.2, 1.0, 2.0]:
            generated = generate_text(
                model, tokenizer, prompt,
                max_tokens=30, temperature=temp, top_p=0.9
            )
            # Clean up for display
            clean_generated = generated.replace('<|endoftext|>', '[END]')[:60]
            print(f"  Temp {temp}: {clean_generated}...")
        
        # Test different top-p values
        for p in [0.5, 0.9, 1.0]:
            generated = generate_text(
                model, tokenizer, prompt,
                max_tokens=30, temperature=1.0, top_p=p
            )
            # Clean up for display  
            clean_generated = generated.replace('<|endoftext|>', '[END]')[:60]
            print(f"  Top-p {p}: {clean_generated}...")
        
        print()
    
    print("Key features demonstrated:")
    print("✓ Temperature scaling: Lower temp → more focused, Higher temp → more random")
    print("✓ Top-p sampling: Dynamically truncates vocabulary based on probability mass")
    print("✓ End-of-text detection: Stops generation when <|endoftext|> is produced")
    print("✓ Max tokens limit: Prevents infinite generation")


def main():
    """Run all demonstrations"""
    print("CS336 Assignment 1: Decoding (3 Points) Implementation")
    print("Section 6: Generating Text")
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demonstrate_temperature_effects()
        demonstrate_top_p_effects() 
        demonstrate_text_generation()
        
        print("=" * 60)
        print("🎉 Decoding Implementation Complete!")
        print("=" * 60)
        print()
        print("This implementation provides:")
        print("1. Temperature scaling: softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σexp(vⱼ/τ)")
        print("2. Top-p (nucleus) sampling for vocabulary truncation")
        print("3. Autoregressive text generation with stopping conditions")
        print("4. Proper handling of context window limits")
        print()
        print("The implementation is ready for use with trained models on")
        print("TinyStories and OpenWebText datasets as specified in the assignment.")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()