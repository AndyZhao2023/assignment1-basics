#!/usr/bin/env python3
"""
Evaluate validation loss for trained models.
Critical for CS336 Assignment 1 deliverable: validation loss ≤ 1.45
"""

import torch
import numpy as np
import json
from pathlib import Path
from cs336_basics.nn import TransformerLM, cross_entropy, get_batch
import argparse

def evaluate_validation_loss(
    model_config_path: str,
    checkpoint_path: str,
    val_data_path: str,
    device: str = 'cpu',
    max_batches: int = 100,
    batch_size: int = 32
):
    """Evaluate model on validation set"""
    
    # Load configuration
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    print(f"Loading model from {checkpoint_path}...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config.get('rope_theta', 10000.0)
    ).to(device)
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading as bare state dict
            try:
                model.load_state_dict(checkpoint)
            except:
                print(f"Warning: Could not load checkpoint from {checkpoint_path}")
                print("Using random initialization - results will not be meaningful!")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found!")
        print("Using random initialization - results will not be meaningful!")
    
    # Load validation data
    print(f"Loading validation data from {val_data_path}...")
    val_data = np.load(val_data_path, mmap_mode='r')
    print(f"Validation set size: {len(val_data):,} tokens")
    
    # Evaluate
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"Evaluating on {max_batches} batches...")
    with torch.no_grad():
        for batch_idx in range(max_batches):
            # Sample random batch from validation set
            try:
                inputs, targets = get_batch(
                    val_data, 
                    batch_size, 
                    config['context_length'], 
                    device
                )
                
                # Forward pass
                logits = model(inputs)
                
                # Calculate loss
                batch_size_actual, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                loss = cross_entropy(logits_flat, targets_flat)
                
                # Accumulate
                total_loss += loss.item() * batch_size_actual * seq_len
                total_tokens += batch_size_actual * seq_len
                
                if (batch_idx + 1) % 10 == 0:
                    current_avg_loss = total_loss / total_tokens
                    print(f"  Batch {batch_idx + 1}/{max_batches}: "
                          f"Avg loss so far: {current_avg_loss:.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Calculate final average
    avg_validation_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    return {
        'validation_loss': avg_validation_loss,
        'perplexity': np.exp(avg_validation_loss),
        'total_tokens_evaluated': total_tokens,
        'num_batches': max_batches
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate validation loss')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (lr_1e-4 or lr_3e-4)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max_batches', type=int, default=100,
                       help='Number of batches to evaluate')
    args = parser.parse_args()
    
    print("=" * 80)
    print("VALIDATION LOSS EVALUATION")
    print("CS336 Assignment 1 - Deliverable Check")
    print("=" * 80)
    
    # Paths based on experiment
    base_dir = Path(f"checkpoints/{args.experiment}")
    log_dir = Path(f"logs/{args.experiment}")
    
    if not base_dir.exists():
        print(f"Error: Checkpoint directory {base_dir} not found!")
        print("Please train the model first.")
        return
    
    # Find latest checkpoint
    checkpoints = list(base_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        print(f"Error: No checkpoints found in {base_dir}")
        return
    
    # Sort by iteration number and get the latest
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    latest_checkpoint = checkpoints[-1]
    
    # Paths
    val_data_path = "artifacts/tinystories_tokens/TinyStoriesV2-GPT4-train_val.npy"
    config_path = log_dir / "artifacts" / "training_config.json"
    
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found!")
        return
    
    print(f"\nEvaluating experiment: {args.experiment}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {latest_checkpoint}")
    print(f"Validation data: {val_data_path}")
    
    # Evaluate
    result = evaluate_validation_loss(
        model_config_path=str(config_path),
        checkpoint_path=str(latest_checkpoint),
        val_data_path=val_data_path,
        device=args.device,
        max_batches=args.max_batches,
        batch_size=args.batch_size
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Validation Loss: {result['validation_loss']:.4f}")
    print(f"Perplexity: {result['perplexity']:.2f}")
    print(f"Tokens Evaluated: {result['total_tokens_evaluated']:,}")
    
    # Check against targets
    gpu_target = 1.45
    cpu_target = 2.00
    
    print("\n" + "=" * 80)
    print("📋 DELIVERABLE STATUS")
    print("=" * 80)
    
    if result['validation_loss'] <= gpu_target:
        print(f"✅ PASS: Validation loss {result['validation_loss']:.4f} ≤ {gpu_target} (GPU target)")
    elif result['validation_loss'] <= cpu_target:
        print(f"✅ PASS: Validation loss {result['validation_loss']:.4f} ≤ {cpu_target} (CPU target)")
    else:
        print(f"❌ FAIL: Validation loss {result['validation_loss']:.4f} > {cpu_target}")
    
    # Save results
    results_file = base_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()