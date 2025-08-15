#!/usr/bin/env python3
"""
CS336 Assignment 1: Complete Training Script
Put it together - comprehensive training pipeline that integrates all components
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cs336_basics.nn import (
    TransformerLM, 
    cross_entropy, 
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer


class TextDataset(Dataset):
    """Memory-mapped text dataset for efficient large-scale training"""
    
    def __init__(self, data_path: str, context_length: int):
        self.context_length = context_length
        
        # Load tokenized data
        if data_path.endswith('.npy'):
            self.data = np.load(data_path, mmap_mode='r')
        else:
            # Assume it's raw text that needs tokenization
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple tokenization by character for demonstration
            self.data = np.array([ord(c) for c in text], dtype=np.int32)
        
        print(f"Loaded dataset with {len(self.data):,} tokens")
    
    def __len__(self):
        return max(0, len(self.data) - self.context_length)
    
    def __getitem__(self, idx):
        # Get a sequence of context_length+1 tokens for input and target
        chunk = self.data[idx:idx + self.context_length + 1]
        return {
            'input_ids': chunk[:-1],
            'targets': chunk[1:]
        }


def train_step(model, batch, optimizer, max_grad_norm: float = 1.0):
    """Single training step with gradient clipping"""
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    targets = batch['targets']
    
    # Forward pass
    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
    
    # Compute loss - reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch_size * seq_len,)
    
    loss = cross_entropy(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    gradient_clipping(model.parameters(), max_grad_norm)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def evaluate_model(model, dataset, batch_size: int, device: str, max_batches: int = 50):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx in range(min(max_batches, len(dataset) // batch_size)):
            # Sample batch
            inputs, targets = get_batch(
                dataset.data, batch_size, dataset.context_length, device
            )
            
            # Forward pass
            logits = model(inputs)
            
            # Compute loss
            batch_size_actual, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=256, 
                       help='Vocabulary size (default: 256 for character-level)')
    parser.add_argument('--context_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0,
                       help='RoPE theta parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Peak learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=3e-5,
                       help='Minimum learning rate for cosine schedule')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay for AdamW')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95,
                       help='AdamW beta2')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Training schedule
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Maximum number of training iterations')
    parser.add_argument('--warmup_iters', type=int, default=100,
                       help='Number of warmup iterations')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='Checkpoint saving interval')
    
    # Data and paths
    parser.add_argument('--data_path', type=str, default='data/sample.txt',
                       help='Path to training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile model with torch.compile')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = TextDataset(args.data_path, args.context_length)
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Compile model if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    total_start_time = time.time()
    losses = []
    
    for iteration in range(start_iter, args.max_iters):
        iter_start_time = time.time()
        
        # Update learning rate
        lr = get_lr_cosine_schedule(
            iteration,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch using the get_batch function
        inputs, targets = get_batch(
            train_dataset.data, args.batch_size, args.context_length, device
        )
        
        # Training step
        batch = {'input_ids': inputs, 'targets': targets}
        loss = train_step(model, batch, optimizer, args.max_grad_norm)
        losses.append(loss)
        
        iter_time = time.time() - iter_start_time
        
        # Logging
        if iteration % 10 == 0 or iteration == args.max_iters - 1:
            avg_loss = np.mean(losses[-100:]) if losses else loss
            print(f"Iter {iteration:6d} | Loss {loss:.4f} | Avg Loss {avg_loss:.4f} | "
                  f"LR {lr:.2e} | Time {iter_time:.3f}s")
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            print("Evaluating...")
            eval_loss = evaluate_model(model, train_dataset, args.batch_size, device)
            print(f"Eval loss: {eval_loss:.4f}")
            model.train()
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:06d}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    
    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss (last 100 iters): {np.mean(losses[-100:]):.4f}")


if __name__ == "__main__":
    main()