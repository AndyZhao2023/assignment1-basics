#!/usr/bin/env python3
"""
Script to train BPE tokenizer on TinyStories dataset for CS336 Assignment 1.
Problem: train_bpe_tinystories (2 points)
"""

import time
import psutil
import os
import json
from pathlib import Path
from cs336_basics.tokenizer import train_bpe

def main():
    # Configuration
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    # Check file size first
    file_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
    print(f"Training BPE tokenizer on {input_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print("-" * 50)
    print("Note: Training on large dataset may take several minutes...")
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Train tokenizer with timing
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    end_time = time.time()
    
    # Get peak memory usage
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate training time
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60
    training_time_hours = training_time_minutes / 60
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_time_hours:.2f} hours ({training_time_minutes:.2f} minutes, {training_time_seconds:.2f} seconds)")
    print(f"Memory usage: {memory_before:.1f} MB -> {memory_after:.1f} MB (Δ {memory_after - memory_before:.1f} MB)")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]
    
    print(f"\nLongest token:")
    print(f"  ID: {longest_token_id}")
    print(f"  Bytes: {longest_token}")
    print(f"  Length: {len(longest_token)} bytes")
    print(f"  As string: {repr(longest_token.decode('utf-8', errors='replace'))}")
    
    # Serialize to disk
    output_dir = Path("tinystories_bpe_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save vocabulary as JSON (convert bytes to string for JSON serialization)
    vocab_json = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    
    # Save merges as text file
    with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
        for merge_a, merge_b in merges:
            f.write(f"{merge_a.decode('utf-8', errors='replace')} {merge_b.decode('utf-8', errors='replace')}\n")
    
    print(f"\nSerialized results to {output_dir}/")
    print(f"  - vocab.json: {len(vocab)} entries")
    print(f"  - merges.txt: {len(merges)} merge rules")
    
    # Answer the assignment questions
    print("\n" + "="*50)
    print("ASSIGNMENT ANSWERS:")
    print("="*50)
    print(f"Training took {training_time_hours:.2f} hours and used approximately {memory_after:.1f} MB of memory.")
    print(f"The longest token is {len(longest_token)} bytes: {repr(longest_token.decode('utf-8', errors='replace'))}, which makes sense as a common phrase in children's stories.")

if __name__ == "__main__":
    main()