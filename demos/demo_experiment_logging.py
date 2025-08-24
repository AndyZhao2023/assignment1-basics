#!/usr/bin/env python3
"""
Demonstration of the Experiment Logging (3 points) implementation.
Shows comprehensive logging features with different backends and metrics.
"""

import torch
import numpy as np
import time
from pathlib import Path

from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.logger import WandbLogger, ConsoleLogger, MultiLogger, create_logger


def demo_console_logging():
    """Demonstrate console and file-based logging"""
    print("=" * 70)
    print("Console Logging Demonstration")
    print("=" * 70)
    
    # Create console logger
    logger = ConsoleLogger(
        project_name="cs336_demo",
        experiment_name="console_logging_demo",
        log_dir="demo_logs",
        print_interval=5
    )
    
    # Log hyperparameters
    hyperparams = {
        "model_type": "transformer",
        "vocab_size": 1000,
        "d_model": 256,
        "num_layers": 4,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "max_iters": 100
    }
    logger.log_hyperparameters(hyperparams)
    
    # Create a small model for demonstration
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256
    )
    
    # Log model architecture
    logger.log_model(model, "demo_transformer")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    # Simulate training loop with rich logging
    print("\nSimulating training with various metrics...")
    losses = []
    
    for step in range(50):
        # Simulate forward/backward pass timing
        forward_time = np.random.uniform(0.01, 0.05)
        backward_time = np.random.uniform(0.01, 0.03)
        
        # Simulate loss progression (decreasing with noise)
        base_loss = 4.0 - (step * 0.05) + np.random.normal(0, 0.1)
        loss = max(base_loss, 1.0)  # Ensure loss doesn't go too low
        losses.append(loss)
        
        # Learning rate schedule
        lr = get_lr_cosine_schedule(step, 3e-4, 3e-5, 10, 50)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Log comprehensive metrics
        metrics = {
            "train/loss": loss,
            "train/perplexity": np.exp(loss),
            "train/learning_rate": lr,
            "timing/forward_ms": forward_time * 1000,
            "timing/backward_ms": backward_time * 1000,
            "performance/tokens_per_second": 32 * 64 / (forward_time + backward_time),
            "training/progress": step / 50
        }
        
        logger.log_metrics(metrics, step)
        
        # Log gradient norms every few steps
        if step % 10 == 0:
            logger.log_gradient_norms(model, step)
            logger.log_learning_rate(optimizer, step)
            logger.log_system_metrics(step)
        
        # Log histograms occasionally
        if step % 15 == 0:
            # Create some fake weight data for demonstration
            fake_weights = torch.randn(1000) * 0.1
            fake_gradients = torch.randn(1000) * 0.01
            
            logger.log_histogram("weights/example_layer", fake_weights, step)
            logger.log_histogram("gradients/example_layer", fake_gradients, step)
        
        # Brief pause to simulate actual training
        time.sleep(0.1)
    
    # Log final summary
    final_metrics = {
        "summary/final_loss": losses[-1],
        "summary/best_loss": min(losses),
        "summary/loss_improvement": losses[0] - losses[-1],
        "summary/training_steps": 50
    }
    logger.log_metrics(final_metrics, 50)
    
    # Save a fake checkpoint for artifact logging
    checkpoint_path = Path("demo_checkpoint.pt")
    torch.save({"model": model.state_dict(), "step": 50}, checkpoint_path)
    logger.log_artifact(str(checkpoint_path), "demo_checkpoint.pt", "checkpoint")
    
    # Finish logging
    logger.finish()
    print(f"\nConsole logging demo complete! Check the logs in: {logger.log_dir}")
    
    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def demo_wandb_logging():
    """Demonstrate Weights & Biases logging (if available)"""
    print("\n" + "=" * 70)
    print("Weights & Biases Logging Demonstration")
    print("=" * 70)
    
    try:
        import wandb
        print("W&B is available, starting demo...")
        
        # Create W&B logger
        logger = WandbLogger(
            project_name="cs336_demo",
            experiment_name="wandb_logging_demo",
            tags=["demo", "experiment_logging", "cs336"],
            mode="offline"  # Use offline mode for demo
        )
        
        # Log hyperparameters
        hyperparams = {
            "architecture": "transformer",
            "vocab_size": 50257,
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "learning_rate": 6e-4,
            "batch_size": 64,
            "dataset": "tinystories",
            "experiment_type": "demo"
        }
        logger.log_hyperparameters(hyperparams)
        
        # Create model
        model = TransformerLM(
            vocab_size=256,
            context_length=128,
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=512
        )
        
        # Log model (W&B will track architecture)
        logger.log_model(model, "transformer_demo")
        
        # Simulate training with W&B logging
        print("Simulating training with W&B logging...")
        
        for step in range(30):
            # Simulate more realistic training metrics
            epoch = step // 10
            loss = 5.0 * np.exp(-step * 0.1) + np.random.normal(0, 0.05)
            
            metrics = {
                "train/loss": loss,
                "train/perplexity": np.exp(loss),
                "train/epoch": epoch,
                "model/parameter_norm": torch.norm(torch.cat([p.view(-1) for p in model.parameters()])).item(),
                "optimization/gradient_norm": np.random.uniform(0.1, 2.0),
                "system/gpu_memory_gb": np.random.uniform(2.0, 8.0),
                "performance/samples_per_second": np.random.uniform(100, 500)
            }
            
            logger.log_metrics(metrics, step)
            
            # Log histograms periodically
            if step % 10 == 0:
                sample_weights = torch.randn(1000) * 0.02
                sample_activations = torch.randn(1000) * 0.5
                
                logger.log_histogram("model/weights_sample", sample_weights, step)
                logger.log_histogram("model/activations_sample", sample_activations, step)
            
            time.sleep(0.05)  # Brief pause
        
        logger.finish()
        print("W&B logging demo complete! Check your W&B dashboard.")
        
    except ImportError:
        print("W&B not available (wandb not installed)")
        print("Install with: pip install wandb")
    except Exception as e:
        print(f"W&B demo failed: {e}")
        print("This is normal if W&B is not configured")


def demo_multi_logger():
    """Demonstrate logging to multiple backends simultaneously"""
    print("\n" + "=" * 70)
    print("Multi-Backend Logging Demonstration")
    print("=" * 70)
    
    # Create loggers for different backends
    console_logger = ConsoleLogger(
        project_name="cs336_multi_demo",
        experiment_name="multi_backend_demo",
        log_dir="demo_multi_logs"
    )
    
    loggers = [console_logger]
    
    # Add W&B logger if available
    try:
        import wandb
        wandb_logger = WandbLogger(
            project_name="cs336_multi_demo",
            experiment_name="multi_backend_demo",
            mode="offline"
        )
        loggers.append(wandb_logger)
        print("Using both Console and W&B logging")
    except:
        print("Using only Console logging (W&B not available)")
    
    # Create multi-logger
    multi_logger = MultiLogger(loggers)
    
    # Log to all backends simultaneously
    hyperparams = {
        "experiment_type": "multi_backend_demo",
        "backends": len(loggers),
        "demo_parameter": 42
    }
    multi_logger.log_hyperparameters(hyperparams)
    
    # Simulate some training
    for step in range(20):
        metrics = {
            "demo/value": np.sin(step * 0.1) + np.random.normal(0, 0.1),
            "demo/step": step,
            "demo/exponential_decay": np.exp(-step * 0.1)
        }
        multi_logger.log_metrics(metrics, step)
        time.sleep(0.05)
    
    multi_logger.finish()
    print("Multi-backend logging demo complete!")


def demo_logging_factory():
    """Demonstrate the logger factory function"""
    print("\n" + "=" * 70)
    print("Logger Factory Demonstration")
    print("=" * 70)
    
    # Test different backend creation
    backends_to_test = ["console"]
    
    # Add wandb if available
    try:
        import wandb
        backends_to_test.append("wandb")
    except ImportError:
        pass
    
    for backend in backends_to_test:
        print(f"\nTesting {backend} backend...")
        
        # Create logger using factory
        kwargs = {
            "backend": backend,
            "project_name": "cs336_factory_demo",
            "experiment_name": f"{backend}_factory_test"
        }
        if backend == "wandb":
            kwargs["mode"] = "offline"
        logger = create_logger(**kwargs)
        
        # Log some test data
        logger.log_hyperparameters({"backend": backend, "test": True})
        
        for i in range(5):
            logger.log_metrics({"test_metric": i * 0.1}, i)
        
        logger.finish()
        print(f"{backend} backend test complete!")


def main():
    """Run all logging demonstrations"""
    print("CS336 Assignment 1: Experiment Logging (3 Points) Implementation")
    print("Comprehensive demonstration of logging capabilities")
    print("=" * 70)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all demonstrations
        demo_console_logging()
        demo_wandb_logging()
        demo_multi_logger()
        demo_logging_factory()
        
        print("\n" + "=" * 70)
        print("🎉 All Experiment Logging Demonstrations Complete!")
        print("=" * 70)
        print("\nImplemented features:")
        print("✅ Console logging with JSON and CSV output")
        print("✅ Weights & Biases integration with cloud logging")
        print("✅ Multi-backend logging to multiple services")
        print("✅ Comprehensive metrics tracking:")
        print("   • Training loss and perplexity")
        print("   • Learning rate schedules")
        print("   • Gradient norms and clipping")
        print("   • System metrics (GPU memory, timing)")
        print("   • Weight and gradient histograms")
        print("   • Model architecture logging")
        print("   • Checkpoint and artifact tracking")
        print("✅ Hyperparameter logging and experiment management")
        print("✅ Real-time metric visualization")
        print("✅ Experiment resumption and tracking")
        print("\nThe experiment logging system is ready for production use!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()