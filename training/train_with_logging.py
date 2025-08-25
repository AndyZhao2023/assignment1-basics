#!/usr/bin/env python3
"""
Enhanced training script with comprehensive experiment logging.
Implements the "Experiment logging (3 points)" problem from CS336 Assignment 1.
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict

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
from cs336_basics.logger import create_logger, ExperimentLogger


class TokenizedDataset(Dataset[Dict[str, np.ndarray]]):
    """Memory-mapped pre-tokenized dataset for efficient large-scale training"""
    
    def __init__(self, data_path: str, context_length: int) -> None:
        self.context_length = context_length
        
        # Load pre-tokenized data (always .npy format)
        self.data = np.load(data_path, mmap_mode='r')
        print(f"Loaded tokenized dataset with {len(self.data):,} tokens from {data_path}")
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Get a sequence of context_length+1 tokens for input and target
        chunk = self.data[idx:idx + self.context_length + 1]
        return {
            'input_ids': chunk[:-1],
            'targets': chunk[1:]
        }


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    logger: Optional[ExperimentLogger] = None,
    step: Optional[int] = None
) -> Dict[str, float]:
    """
    Single training step with gradient clipping and logging.
    
    Returns:
        Dictionary of metrics from this step
    """
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    targets = batch['targets']
    
    # Forward pass
    forward_start = time.time()
    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
    forward_time = time.time() - forward_start
    
    # Compute loss - reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    loss = cross_entropy(logits_flat, targets_flat)
    
    # Backward pass
    backward_start = time.time()
    loss.backward()
    backward_time = time.time() - backward_start
    
    # Calculate gradient norm before clipping
    total_norm_before = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2).item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    # Gradient clipping
    gradient_clipping(model.parameters(), max_grad_norm)
    
    # Calculate gradient norm after clipping
    total_norm_after = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2).item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    # Optimizer step
    optimizer_start = time.time()
    optimizer.step()
    optimizer_time = time.time() - optimizer_start
    
    # Prepare metrics
    metrics = {
        "train/loss": loss.item(),
        "train/perplexity": np.exp(loss.item()),
        "train/grad_norm_before_clip": total_norm_before,
        "train/grad_norm_after_clip": total_norm_after,
        "train/grad_clip_ratio": total_norm_after / (total_norm_before + 1e-8),
        "timing/forward_ms": forward_time * 1000,
        "timing/backward_ms": backward_time * 1000,
        "timing/optimizer_ms": optimizer_time * 1000,
        "timing/total_ms": (forward_time + backward_time + optimizer_time) * 1000,
    }
    
    # Log metrics if logger provided
    if logger:
        logger.log_metrics(metrics, step)
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    val_dataset: TokenizedDataset,
    batch_size: int,
    device: str,
    max_batches: int = 50,
    logger: Optional[ExperimentLogger] = None,
    step: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation dataset.
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx in range(min(max_batches, len(val_dataset) // batch_size)):
            # Sample batch from validation data
            inputs, targets = get_batch(
                val_dataset.data, batch_size, val_dataset.context_length, device
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
    
    eval_time = time.time() - eval_start
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    metrics = {
        "eval/loss": avg_loss,
        "eval/perplexity": np.exp(avg_loss),
        "eval/num_batches": num_batches,
        "timing/eval_total_seconds": eval_time
    }
    
    if logger:
        logger.log_metrics(metrics, step)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model with Experiment Logging')
    
    # Model hyperparameters (PDF-specified for 17M parameter model)
    parser.add_argument('--vocab_size', type=int, default=10000, 
                       help='Vocabulary size (10K BPE tokens from PDF)')
    parser.add_argument('--context_length', type=int, default=256,
                       help='Maximum sequence length (PDF specification)')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension (PDF specification)')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers (PDF specification)')
    parser.add_argument('--num_heads', type=int, default=16,
                       help='Number of attention heads (PDF specification)')
    parser.add_argument('--d_ff', type=int, default=1344,
                       help='Feed-forward dimension (PDF specification)')
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
    parser.add_argument('--validation_batches', type=int, default=50,
                       help='Number of batches to use for validation')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='cs336-training',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity (username or team)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment run')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='How often to log metrics')
    parser.add_argument('--log_histograms', action='store_true',
                       help='Log weight and gradient histograms')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for local logging')
    
    # Data and paths
    parser.add_argument('--data_path', type=str, default='artifacts/tinystories_tokens/TinyStoriesV2-GPT4-train_train.npy',
                       help='Path to pre-tokenized training data (.npy file)')
    parser.add_argument('--val_data_path', type=str, default='artifacts/tinystories_tokens/TinyStoriesV2-GPT4-train_val.npy',
                       help='Path to pre-tokenized validation data (.npy file)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile model with torch.compile')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    
    # Create logger
    if args.use_wandb:
        logger = create_logger(
            backend="wandb",
            project_name=args.wandb_project,
            experiment_name=args.experiment_name,
            entity=args.wandb_entity,
            tags=["transformer", "language-model", "cs336"],
            mode="online"
        )
    else:
        logger = create_logger(
            backend="console",
            project_name="cs336-training",
            experiment_name=args.experiment_name,
            log_dir=args.log_dir,
            print_interval=args.log_interval
        )
    
    # Log hyperparameters
    hyperparameters = {
        # Model architecture
        "model/vocab_size": args.vocab_size,
        "model/context_length": args.context_length,
        "model/d_model": args.d_model,
        "model/num_layers": args.num_layers,
        "model/num_heads": args.num_heads,
        "model/d_ff": args.d_ff,
        "model/rope_theta": args.rope_theta,
        
        # Training
        "training/batch_size": args.batch_size,
        "training/learning_rate": args.learning_rate,
        "training/min_learning_rate": args.min_learning_rate,
        "training/weight_decay": args.weight_decay,
        "training/beta1": args.beta1,
        "training/beta2": args.beta2,
        "training/max_grad_norm": args.max_grad_norm,
        "training/max_iters": args.max_iters,
        "training/warmup_iters": args.warmup_iters,
        
        # System
        "system/device": device,
        "system/compile": args.compile,
        "system/seed": args.seed,
        
        # Data
        "data/train_path": args.data_path,
        "data/val_path": args.val_data_path,
    }
    logger.log_hyperparameters(hyperparameters)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Log config as artifact
    logger.log_artifact(str(config_path), "training_config.json", "config")
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = TokenizedDataset(args.data_path, args.context_length)
    
    print("Loading validation dataset...")
    val_dataset = TokenizedDataset(args.val_data_path, args.context_length)
    
    # Initialize model
    print("Initializing model...")
    model: torch.nn.Module = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model has {num_params:,} parameters ({num_trainable:,} trainable)")
    
    # Log model architecture
    logger.log_model(model, "transformer_lm")
    logger.log_metrics({
        "model/total_parameters": num_params,
        "model/trainable_parameters": num_trainable
    }, step=0)
    
    # Compile model if requested
    if args.compile:
        print("Compiling model...")
        if device == 'mps':
            # MPS needs special backend according to PDF tips
            model = torch.compile(model, backend="aot_eager")  # type: ignore[assignment]
        else:
            # CPU and CUDA can use default backend
            model = torch.compile(model)  # type: ignore[assignment]
    
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
        start_iter = load_checkpoint(args.resume_from, model, optimizer)  # type: ignore[arg-type]
        print(f"Resumed from iteration {start_iter}")
        logger.log_metrics({"training/resumed_from_iteration": start_iter}, step=start_iter)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    total_start_time = time.time()
    running_loss = 0.0
    loss_window = []
    
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
        
        # Log learning rate
        if iteration % args.log_interval == 0:
            logger.log_learning_rate(optimizer, iteration)
        
        # Get batch
        inputs, targets = get_batch(
            train_dataset.data, args.batch_size, args.context_length, device
        )
        batch = {'input_ids': inputs, 'targets': targets}
        
        # Training step with logging
        metrics = train_step(
            model, batch, optimizer, args.max_grad_norm,
            logger if iteration % args.log_interval == 0 else None,
            iteration
        )
        
        # Track loss for averaging
        loss = metrics["train/loss"]
        loss_window.append(loss)
        if len(loss_window) > 100:
            loss_window.pop(0)
        running_loss = np.mean(loss_window)
        
        # Log additional metrics
        if iteration % args.log_interval == 0:
            iter_time = time.time() - iter_start_time
            tokens_per_sec = args.batch_size * args.context_length / iter_time
            
            additional_metrics = {
                "train/running_loss": running_loss,
                "train/learning_rate": lr,
                "performance/tokens_per_second": tokens_per_sec,
                "performance/iterations_per_second": 1.0 / iter_time,
                "training/iteration": iteration,
                "training/progress": iteration / args.max_iters
            }
            logger.log_metrics(additional_metrics, iteration)
            
            # Log system metrics
            logger.log_system_metrics(iteration)
            
            # Log histograms if requested
            if args.log_histograms and iteration % (args.log_interval * 10) == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.log_histogram(f"weights/{name}", param.data, iteration)
                        if param.grad is not None:
                            logger.log_histogram(f"gradients/{name}", param.grad, iteration)
        
        # Console output
        if iteration % 10 == 0 or iteration == args.max_iters - 1:
            iter_time = time.time() - iter_start_time
            print(f"Iter {iteration:6d}/{args.max_iters} | "
                  f"Loss {loss:.4f} | Avg {running_loss:.4f} | "
                  f"LR {lr:.2e} | Time {iter_time:.3f}s")
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            print("Evaluating...")
            eval_metrics = evaluate_model(
                model, val_dataset, args.batch_size, device,
                max_batches=args.validation_batches, logger=logger, step=iteration
            )
            print(f"Eval loss: {eval_metrics['eval/loss']:.4f} | "
                  f"Eval perplexity: {eval_metrics['eval/perplexity']:.2f}")
            model.train()
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:06d}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)  # type: ignore[arg-type]
            
            # Log checkpoint as artifact
            logger.log_artifact(str(checkpoint_path), f"checkpoint_{iteration:06d}.pt", "checkpoint")
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)  # type: ignore[arg-type]
    logger.log_artifact(str(final_checkpoint_path), "checkpoint_final.pt", "checkpoint")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval_metrics = evaluate_model(
        model, val_dataset, args.batch_size, device,
        max_batches=args.validation_batches * 2, logger=logger, step=args.max_iters
    )
    
    # Training summary
    total_time = time.time() - total_start_time
    summary_metrics = {
        "summary/total_training_time_hours": total_time / 3600,
        "summary/total_iterations": args.max_iters - start_iter,
        "summary/final_train_loss": loss,
        "summary/final_eval_loss": final_eval_metrics["eval/loss"],
        "summary/final_eval_perplexity": final_eval_metrics["eval/perplexity"],
        "summary/average_tokens_per_second": (args.max_iters - start_iter) * args.batch_size * args.context_length / total_time
    }
    logger.log_metrics(summary_metrics, args.max_iters)
    
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Final train loss: {loss:.4f}")
    print(f"Final eval loss: {final_eval_metrics['eval/loss']:.4f}")
    print(f"Final eval perplexity: {final_eval_metrics['eval/perplexity']:.2f}")
    
    # Finish logging
    logger.finish()
    print("\nExperiment logging complete!")


if __name__ == "__main__":
    main()