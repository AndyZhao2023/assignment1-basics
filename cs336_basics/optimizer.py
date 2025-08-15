from __future__ import annotations

import torch
import math
from collections.abc import Iterable


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer implementation based on Loshchilov and Hutter [2019].
    
    Implements adaptive learning rates with decoupled weight decay.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients used for computing running averages of gradient
                   and its square (default: (0.9, 0.999))
            eps: Term added to denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 1e-2)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: callable | None = None) -> float | None:
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            The loss if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Get parameter state
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute bias-corrected learning rate
                corrected_lr = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-corrected_lr)
                
                # Apply weight decay (decoupled)
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
        
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (reached after warmup)
        min_learning_rate: Minimum learning rate (final value)
        warmup_iters: Number of iterations for linear warmup
        cosine_cycle_iters: Total number of iterations for the cosine cycle
        
    Returns:
        Learning rate for the given iteration
    """
    if it < warmup_iters:
        # Linear warmup phase
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        # Cosine decay phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    else:
        # Post-cycle: constant at minimum
        return min_learning_rate