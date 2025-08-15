#!/usr/bin/env python3
"""
Example: Learning Rate Scheduler Integration with AdamW and Gradient Clipping

This demonstrates how to use all three components together:
1. AdamW optimizer (our implementation)
2. Cosine learning rate schedule with warmup (our implementation)  
3. Gradient clipping (our implementation)
"""

import torch
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn import gradient_clipping


def simple_training_example():
    """Minimal example showing scheduler integration."""
    
    # Create a simple model for demonstration
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    # Training hyperparameters
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 50
    total_iters = 500
    max_grad_norm = 1.0
    
    # Initialize our AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,  # Will be overridden by scheduler
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    print("Training with Learning Rate Scheduler")
    print("=" * 50)
    print(f"Max LR: {max_lr:.2e}, Min LR: {min_lr:.2e}")
    print(f"Warmup: {warmup_iters} steps, Total: {total_iters} steps")
    print("=" * 50)
    
    # Training loop
    model.train()
    for iteration in range(total_iters):
        
        # === 1. UPDATE LEARNING RATE ===
        current_lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=total_iters
        )
        
        # Apply to all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # === 2. TRAINING STEP ===
        # Create dummy batch
        batch_x = torch.randn(32, 100)
        batch_y = torch.randint(0, 10, (32,))
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = torch.nn.functional.cross_entropy(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # === 3. GRADIENT CLIPPING ===
        gradient_clipping(model.parameters(), max_grad_norm)
        
        # === 4. OPTIMIZER STEP ===
        optimizer.step()
        
        # === 5. LOGGING ===
        if iteration % 50 == 0 or iteration < 10:
            # Check what phase we're in
            if iteration < warmup_iters:
                phase = "WARMUP"
            elif iteration < total_iters:
                phase = "COSINE"
            else:
                phase = "POST"
                
            print(f"Step {iteration:3d} | {phase:6s} | LR: {current_lr:.2e} | Loss: {loss:.4f}")
    
    print("=" * 50)
    print("Training completed!")
    

def demonstrate_lr_curve():
    """Show the learning rate curve in detail."""
    
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 20
    total_iters = 100
    
    print("\\nDetailed Learning Rate Schedule:")
    print("=" * 40)
    
    for i in range(0, total_iters + 10, 10):
        lr = get_lr_cosine_schedule(i, max_lr, min_lr, warmup_iters, total_iters)
        
        # Create visual representation
        bar_length = 30
        if i < warmup_iters:
            # Warmup phase - show progress
            progress = i / warmup_iters
            filled = int(progress * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            phase = "WARMUP"
        elif i < total_iters:
            # Cosine phase - show decay
            progress = (i - warmup_iters) / (total_iters - warmup_iters)
            filled = int((1 - progress) * bar_length)  # Inverse for decay
            bar = "█" * filled + "░" * (bar_length - filled)
            phase = "COSINE"
        else:
            # Post phase - constant
            bar = "░" * bar_length
            phase = "POST"
        
        print(f"Step {i:3d} | {phase:6s} | {bar} | LR: {lr:.2e}")


if __name__ == "__main__":
    # Run the training example
    simple_training_example()
    
    # Show detailed LR curve
    demonstrate_lr_curve()
    
    print("\\n🎯 Key Takeaways:")
    print("  1. Learning rate is updated EVERY iteration")
    print("  2. Warmup prevents early training instability") 
    print("  3. Cosine decay provides smooth convergence")
    print("  4. Gradient clipping works with any learning rate")
    print("  5. All three components integrate seamlessly!")