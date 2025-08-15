# How Learning Rate Schedulers Are Used in Training

## Overview

Learning rate schedulers dynamically adjust the learning rate during training to improve convergence and stability. Our `get_lr_cosine_schedule()` function implements the cosine annealing schedule with warmup, commonly used in transformer training.

## Complete Training Example

```python
import torch
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn import gradient_clipping, TransformerLM

def train_transformer():
    # 1. Setup model and training parameters
    model = TransformerLM(
        vocab_size=50257,
        context_length=1024, 
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000
    )
    
    # 2. Training hyperparameters
    max_learning_rate = 6e-4     # Peak learning rate
    min_learning_rate = 6e-5     # Final learning rate (10% of max)
    warmup_iters = 2000          # Warmup steps (2000 iterations)
    total_iters = 100000         # Total training iterations
    max_grad_norm = 1.0          # Gradient clipping threshold
    
    # 3. Initialize optimizer with max learning rate
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,      # This will be overridden by scheduler
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )
    
    # 4. Training loop
    model.train()
    for iteration in range(total_iters):
        
        # === LEARNING RATE SCHEDULING ===
        # Get current learning rate from our scheduler
        current_lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=total_iters
        )
        
        # Update all parameter groups in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # === TRAINING STEP ===
        # Get batch (implement your data loading here)
        batch = get_training_batch()  # Shape: [batch_size, seq_len]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch)
        loss = compute_loss(logits, batch)  # Your loss function
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (using our implementation!)
        gradient_clipping(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # === LOGGING ===
        if iteration % 500 == 0:
            print(f"Iter {iteration:6d} | LR: {current_lr:.2e} | Loss: {loss:.4f}")
            
        # === PHASE TRACKING ===
        if iteration == warmup_iters:
            print(f"✅ Warmup complete at iteration {iteration}")
        elif iteration == total_iters - 1:
            print(f"✅ Training complete at iteration {iteration}")

def get_training_batch():
    """Placeholder for your data loading logic"""
    batch_size = 32
    seq_len = 1024
    vocab_size = 50257
    return torch.randint(0, vocab_size, (batch_size, seq_len))

def compute_loss(logits, targets):
    """Placeholder for your loss computation"""
    # Example: next-token prediction loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

if __name__ == "__main__":
    train_transformer()
```

## Learning Rate Schedule Phases

### Phase 1: Linear Warmup (0 → warmup_iters)
```python
# Learning rate increases linearly from 0 to max_learning_rate
lr = max_learning_rate * (iteration / warmup_iters)

# Example with warmup_iters=2000, max_lr=6e-4:
# Iteration 0:    LR = 6e-4 * (0/2000)    = 0.0
# Iteration 500:  LR = 6e-4 * (500/2000)  = 1.5e-4  
# Iteration 1000: LR = 6e-4 * (1000/2000) = 3.0e-4
# Iteration 2000: LR = 6e-4 * (2000/2000) = 6e-4
```

### Phase 2: Cosine Decay (warmup_iters → total_iters)
```python
# Learning rate follows cosine curve from max to min
progress = (iteration - warmup_iters) / (total_iters - warmup_iters)
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

# Smooth decay from 6e-4 down to 6e-5 over ~98,000 iterations
```

### Phase 3: Post-Training (> total_iters)
```python
# Learning rate stays constant at minimum value
lr = min_learning_rate  # 6e-5
```

## Integration with PyTorch's Built-in Schedulers

You can also wrap our function to work with PyTorch's scheduler interface:

```python
class CosineSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, total_iters):
        self.max_lr = max_lr
        self.min_lr = min_lr  
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, iteration):
        lr = get_lr_cosine_schedule(
            iteration, self.max_lr, self.min_lr, 
            self.warmup_iters, self.total_iters
        )
        return lr / self.max_lr  # LambdaLR expects multiplier

# Usage with PyTorch scheduler
optimizer = AdamW(model.parameters(), lr=max_learning_rate)
scheduler = CosineSchedulerWithWarmup(optimizer, max_lr, min_lr, warmup_iters, total_iters)

for iteration in range(total_iters):
    # Training step
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    optimizer.step()
    
    # Update learning rate
    scheduler.step()  # Calls our function internally
```

## Best Practices

### 1. **Choosing Scheduler Parameters**
```python
# Common ratios for transformer training:
warmup_iters = total_iters * 0.01    # 1% of training for warmup
min_lr = max_lr * 0.1                # Final LR is 10% of peak LR

# For GPT-style training:
max_learning_rate = 6e-4             # Proven good for transformers
min_learning_rate = 6e-5             # Don't go too low
warmup_iters = 2000                  # 2000 steps works well
```

### 2. **Scheduler + Gradient Clipping**
```python
# ALWAYS do gradient clipping BEFORE optimizer.step()
loss.backward()
gradient_clipping(model.parameters(), max_norm=1.0)  # Clip first
optimizer.step()                                     # Then update
```

### 3. **Monitoring Learning Rate**
```python
# Log learning rate to track schedule
if iteration % 100 == 0:
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr, "iteration": iteration})
```

### 4. **Resuming Training**
```python
# When resuming from checkpoint, make sure to restore LR correctly
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Resume scheduler from correct iteration
start_iteration = checkpoint['iteration']
for iteration in range(start_iteration, total_iters):
    current_lr = get_lr_cosine_schedule(iteration, ...)  # Use actual iteration
    # ... rest of training
```

## Why This Schedule Works

1. **Warmup prevents early instability**: Large learning rates at start can cause exploding gradients
2. **Cosine decay is smooth**: No sudden LR drops that could hurt convergence  
3. **Low final LR enables fine-tuning**: Model can make small adjustments near convergence
4. **Widely validated**: Used successfully in GPT, BERT, and other large models

This scheduler is essential for stable transformer training and works seamlessly with our AdamW optimizer and gradient clipping implementations!