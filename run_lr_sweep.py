#!/usr/bin/env python3
"""
Learning Rate Sweep for CS336 Assignment 1
Target: Find optimal learning rate to achieve validation loss ≤ 2.00
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

def format_lr(lr: float) -> str:
    """Format learning rate for directory names"""
    if lr >= 1e-2:
        return f"{lr:.2f}".replace(".", "_")
    else:
        return f"{lr:.0e}".replace(".", "_").replace("+", "").replace("-", "m")

def run_training(
    lr: float,
    max_iters: int = 5000,
    quick_test: bool = False,
    device: str = "cpu"
) -> Dict:
    """Run training with specified learning rate"""
    
    lr_str = format_lr(lr)
    
    if quick_test:
        max_iters = 500  # Quick divergence test
        eval_interval = 50
        save_interval = 500
        print(f"\n🔍 Quick divergence test for LR={lr}")
    else:
        eval_interval = 250
        save_interval = 1000
        print(f"\n🚀 Full training run for LR={lr}")
    
    # Calculate min_lr as 1/10 of the main lr (for cosine schedule)
    min_lr = lr / 10
    
    cmd = [
        'uv', 'run', 'python', '-m', 'training.train_with_logging',
        '--data_path', 'artifacts/tinystories_tokens/TinyStoriesV2-GPT4-train_train.npy',
        '--val_data_path', 'artifacts/tinystories_tokens/TinyStoriesV2-GPT4-train_val.npy',
        '--vocab_size', '10000',
        '--context_length', '256',  # Reduced for CPU
        '--d_model', '512',
        '--num_layers', '4',
        '--num_heads', '16',
        '--d_ff', '1344',
        '--learning_rate', str(lr),
        '--min_learning_rate', str(min_lr),
        '--batch_size', '32',
        '--device', device,
        '--max_iters', str(max_iters),
        '--warmup_iters', str(max_iters // 10),  # 10% warmup
        '--eval_interval', str(eval_interval),
        '--save_interval', str(save_interval),
        '--log_dir', f'logs/lr_sweep/lr_{lr_str}',
        '--checkpoint_dir', f'checkpoints/lr_sweep/lr_{lr_str}',
        '--seed', '42',
        '--beta1', '0.9',
        '--beta2', '0.95',
        '--weight_decay', '0.1'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Expected tokens: {32 * max_iters * 256:,}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        # Parse output for final metrics
        lines = result.stdout.split('\n')
        final_train_loss = None
        final_val_loss = None
        diverged = False
        
        for line in reversed(lines):
            if 'val_loss' in line and final_val_loss is None:
                try:
                    # Extract validation loss from output
                    parts = line.split('val_loss:')
                    if len(parts) > 1:
                        val_loss_str = parts[1].split()[0].strip(',')
                        final_val_loss = float(val_loss_str)
                except:
                    pass
            
            if 'train_loss' in line and final_train_loss is None:
                try:
                    parts = line.split('train_loss:')
                    if len(parts) > 1:
                        train_loss_str = parts[1].split()[0].strip(',')
                        final_train_loss = float(train_loss_str)
                        # Check for divergence (NaN or very high loss)
                        if train_loss_str == 'nan' or final_train_loss > 100:
                            diverged = True
                except:
                    pass
        
        # Check for divergence indicators in stderr
        if 'nan' in result.stderr.lower() or 'inf' in result.stderr.lower():
            diverged = True
        
        success = result.returncode == 0 and not diverged
        
        return {
            'lr': lr,
            'lr_str': lr_str,
            'success': success,
            'diverged': diverged,
            'duration': duration,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'max_iters': max_iters,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out for LR={lr}")
        return {
            'lr': lr,
            'lr_str': lr_str,
            'success': False,
            'diverged': False,
            'error': 'timeout',
            'max_iters': max_iters
        }
    except Exception as e:
        print(f"❌ Error running training for LR={lr}: {e}")
        return {
            'lr': lr,
            'lr_str': lr_str,
            'success': False,
            'diverged': False,
            'error': str(e),
            'max_iters': max_iters
        }

def main():
    print("="*80)
    print("CS336 Assignment 1: Learning Rate Sweep")
    print("Target: Validation loss ≤ 2.00 (CPU/MPS)")
    print("="*80)
    
    # Learning rates to test
    # Start with key rates, then add more based on results
    learning_rates = [
        (1e-4, False),   # Medium-low (should be stable)
        (3e-4, False),   # Default from literature
        (1e-3, False),   # High rate
        (3e-3, True),    # Very high (quick test for divergence)
    ]
    
    results = []
    
    print(f"\nPlanning to test {len(learning_rates)} learning rates")
    print("Learning rates:", [lr for lr, _ in learning_rates])
    
    # Run experiments
    for lr, quick_test in learning_rates:
        print("\n" + "="*60)
        result = run_training(lr, quick_test=quick_test)
        results.append(result)
        
        # Summary for this run
        if result['success']:
            print(f"✅ LR={lr}: Success!")
            if result['final_val_loss']:
                print(f"   Final validation loss: {result['final_val_loss']:.4f}")
                if result['final_val_loss'] <= 2.00:
                    print(f"   🎯 MEETS TARGET (≤ 2.00)")
        elif result['diverged']:
            print(f"💥 LR={lr}: DIVERGED (edge of stability found)")
        else:
            print(f"❌ LR={lr}: Failed")
        
        print(f"   Duration: {result.get('duration', 0)/60:.1f} minutes")
    
    # Save results
    results_file = Path("logs/lr_sweep/sweep_results.json")
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    
    # Find best result
    successful_results = [r for r in results if r['success'] and r.get('final_val_loss')]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_val_loss'])
        print(f"🏆 Best learning rate: {best_result['lr']}")
        print(f"   Validation loss: {best_result['final_val_loss']:.4f}")
        
        if best_result['final_val_loss'] <= 2.00:
            print(f"✅ TARGET ACHIEVED! Validation loss ≤ 2.00")
        else:
            print(f"⚠️  Best loss {best_result['final_val_loss']:.4f} > 2.00 target")
    
    # Report divergence
    diverged_results = [r for r in results if r.get('diverged')]
    if diverged_results:
        min_diverged_lr = min(r['lr'] for r in diverged_results)
        print(f"\n🔍 Edge of stability: LR ≥ {min_diverged_lr} causes divergence")
    
    print(f"\nResults saved to: {results_file}")
    print("\nNext steps:")
    print("1. Run evaluate_validation.py on best model")
    print("2. Generate learning curves visualization")
    print("3. Analyze relationship between optimal LR and stability edge")

if __name__ == "__main__":
    main()