"""
CS336 Assignment 1: Transformer LM Resource Accounting

This module provides functions to calculate the resource requirements 
(parameters, memory, FLOPs) for GPT-2 style Transformer language models.

Problem: transformer_accounting (5 points)
"""

import math
from typing import Dict, Tuple


def count_parameters(vocab_size: int, context_length: int, d_model: int, 
                    num_layers: int, num_heads: int, d_ff: int) -> Dict[str, int]:
    """
    Count the number of trainable parameters for each component of a GPT-2 style Transformer LM.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension  
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        
    Returns:
        Dictionary with parameter counts for each component and total
    """
    
    # Token embeddings: vocab_size × d_model
    token_embeddings = vocab_size * d_model
    
    # Per-layer components
    # Multi-head self-attention: Q, K, V projections + output projection
    # Each projection is d_model × d_model
    attention_per_layer = 4 * (d_model * d_model)
    
    # Layer normalization (RMSNorm): d_model parameters per layer (2 layers per block)
    layer_norm_per_layer = 2 * d_model
    
    # Feed-forward network (SwiGLU): w1, w2, w3
    # w1: d_model × d_ff, w2: d_ff × d_model, w3: d_model × d_ff
    ffn_per_layer = d_model * d_ff + d_ff * d_model + d_model * d_ff
    ffn_per_layer = d_model * d_ff * 3  # Simplified
    
    # Total per layer
    params_per_layer = attention_per_layer + layer_norm_per_layer + ffn_per_layer
    
    # All layers
    all_layers = num_layers * params_per_layer
    
    # Final layer norm
    final_layer_norm = d_model
    
    # Language model head (output projection): d_model × vocab_size
    lm_head = d_model * vocab_size
    
    # Total parameters
    total_params = token_embeddings + all_layers + final_layer_norm + lm_head
    
    return {
        'token_embeddings': token_embeddings,
        'attention_per_layer': attention_per_layer,
        'layer_norm_per_layer': layer_norm_per_layer, 
        'ffn_per_layer': ffn_per_layer,
        'params_per_layer': params_per_layer,
        'all_layers': all_layers,
        'final_layer_norm': final_layer_norm,
        'lm_head': lm_head,
        'total': total_params
    }


def calculate_memory_usage(param_count: int, batch_size: int, sequence_length: int, 
                          d_model: int, num_layers: int, precision_bytes: int = 4) -> Dict[str, float]:
    """
    Calculate memory usage in GB for parameters, activations, gradients, and optimizer state.
    
    Args:
        param_count: Total number of parameters
        batch_size: Batch size
        sequence_length: Sequence length
        d_model: Model dimension
        num_layers: Number of layers
        precision_bytes: Bytes per parameter (4 for float32, 2 for float16)
        
    Returns:
        Dictionary with memory usage in GB for each component
    """
    
    # Parameters memory
    params_memory = param_count * precision_bytes
    
    # Activations memory (rough estimate)
    # For each layer, we need to store intermediate activations
    # Main activations: embeddings, attention outputs, FFN outputs
    activations_per_layer = batch_size * sequence_length * d_model
    total_activations = activations_per_layer * num_layers * 2  # 2 for attention + FFN
    activations_memory = total_activations * precision_bytes
    
    # Gradients memory (same size as parameters)
    gradients_memory = params_memory
    
    # AdamW optimizer state: 2 additional copies of parameters (momentum and variance)
    optimizer_memory = 2 * params_memory
    
    # Convert to GB
    def bytes_to_gb(bytes_val):
        return bytes_val / (1024**3)
    
    return {
        'parameters_gb': bytes_to_gb(params_memory),
        'activations_gb': bytes_to_gb(activations_memory),
        'gradients_gb': bytes_to_gb(gradients_memory),
        'optimizer_gb': bytes_to_gb(optimizer_memory),
        'total_gb': bytes_to_gb(params_memory + activations_memory + gradients_memory + optimizer_memory)
    }


def calculate_flops_per_forward_pass(batch_size: int, sequence_length: int, 
                                   d_model: int, num_layers: int, num_heads: int, 
                                   d_ff: int, vocab_size: int) -> Dict[str, int]:
    """
    Calculate FLOPs for a single forward pass through the transformer.
    
    Matrix multiplication FLOPs: For A ∈ R^(m×n) × B ∈ R^(n×p), FLOPs = 2mnp
    
    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        d_model: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        
    Returns:
        Dictionary with FLOPs for each component
    """
    
    # Token embeddings: no matrix multiply, just lookup
    token_embedding_flops = 0
    
    # Per-layer FLOPs
    d_k = d_model // num_heads
    
    # Attention FLOPs per layer
    # Q, K, V projections: 3 × (batch_size × seq_len × d_model) × d_model
    qkv_proj_flops = 3 * (2 * batch_size * sequence_length * d_model * d_model)
    
    # Attention scores: Q @ K^T for each head
    # (batch_size × num_heads × seq_len × d_k) @ (batch_size × num_heads × d_k × seq_len)
    attention_scores_flops = 2 * batch_size * num_heads * sequence_length * sequence_length * d_k
    
    # Attention output: scores @ V for each head  
    # (batch_size × num_heads × seq_len × seq_len) @ (batch_size × num_heads × seq_len × d_k)
    attention_output_flops = 2 * batch_size * num_heads * sequence_length * sequence_length * d_k
    
    # Output projection: (batch_size × seq_len × d_model) @ (d_model × d_model)
    output_proj_flops = 2 * batch_size * sequence_length * d_model * d_model
    
    attention_flops_per_layer = qkv_proj_flops + attention_scores_flops + attention_output_flops + output_proj_flops
    
    # Feed-forward FLOPs per layer (SwiGLU)
    # w1 projection: (batch_size × seq_len × d_model) @ (d_model × d_ff)
    w1_flops = 2 * batch_size * sequence_length * d_model * d_ff
    
    # w3 projection: (batch_size × seq_len × d_model) @ (d_model × d_ff) 
    w3_flops = 2 * batch_size * sequence_length * d_model * d_ff
    
    # w2 projection: (batch_size × seq_len × d_ff) @ (d_ff × d_model)
    w2_flops = 2 * batch_size * sequence_length * d_ff * d_model
    
    ffn_flops_per_layer = w1_flops + w3_flops + w2_flops
    
    # Total per layer
    flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
    
    # All layers
    all_layers_flops = num_layers * flops_per_layer
    
    # Language model head: (batch_size × seq_len × d_model) @ (d_model × vocab_size)
    lm_head_flops = 2 * batch_size * sequence_length * d_model * vocab_size
    
    # Total FLOPs
    total_flops = token_embedding_flops + all_layers_flops + lm_head_flops
    
    return {
        'token_embeddings': token_embedding_flops,
        'qkv_projections_per_layer': qkv_proj_flops,
        'attention_scores_per_layer': attention_scores_flops,
        'attention_output_per_layer': attention_output_flops,
        'output_projection_per_layer': output_proj_flops,
        'attention_total_per_layer': attention_flops_per_layer,
        'ffn_w1_per_layer': w1_flops,
        'ffn_w3_per_layer': w3_flops,
        'ffn_w2_per_layer': w2_flops,
        'ffn_total_per_layer': ffn_flops_per_layer,
        'total_per_layer': flops_per_layer,
        'all_layers': all_layers_flops,
        'lm_head': lm_head_flops,
        'total': total_flops
    }


def gpt2_xl_analysis() -> None:
    """
    Part (a): Analyze GPT-2 XL configuration.
    
    GPT-2 XL configuration:
    - vocab_size = 50257
    - context_length = 1024
    - d_model = 1600 
    - num_layers = 48
    - num_heads = 25
    - d_ff = 6400
    """
    print("=== GPT-2 XL Resource Accounting ===\n")
    
    # Configuration
    vocab_size = 50257
    context_length = 1024
    d_model = 1600
    num_layers = 48
    num_heads = 25
    d_ff = 6400
    
    print(f"Configuration:")
    print(f"  vocab_size = {vocab_size:,}")
    print(f"  context_length = {context_length}")
    print(f"  d_model = {d_model}")
    print(f"  num_layers = {num_layers}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff:,}")
    print()
    
    # Part (a): Parameter counting
    print("Part (a): Parameter Counting")
    params = count_parameters(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    
    print(f"Token embeddings: {params['token_embeddings']:,} parameters")
    print(f"Per layer:")
    print(f"  - Attention: {params['attention_per_layer']:,} parameters")
    print(f"  - Layer norm: {params['layer_norm_per_layer']:,} parameters")
    print(f"  - Feed-forward: {params['ffn_per_layer']:,} parameters")
    print(f"  - Total per layer: {params['params_per_layer']:,} parameters")
    print(f"All {num_layers} layers: {params['all_layers']:,} parameters")
    print(f"Final layer norm: {params['final_layer_norm']:,} parameters")
    print(f"LM head: {params['lm_head']:,} parameters")
    print(f"TOTAL PARAMETERS: {params['total']:,} parameters ({params['total']/1e9:.2f}B)")
    print()
    
    # Memory usage analysis for model parameters (as specified in part a)
    print("Memory Usage Analysis (single-precision floating point):")
    params_memory_bytes = params['total'] * 4  # 4 bytes per float32 parameter
    params_memory_gb = params_memory_bytes / (1024**3)
    
    print(f"Model parameters: {params_memory_gb:.2f} GB")
    print(f"Total parameters: {params['total']:,} parameters")
    print(f"Memory per parameter: 4 bytes (float32)")
    print()
    
    # Additional analysis for training memory (example with batch_size=1, seq_len=1024)
    print("Additional Training Memory Analysis (example: batch_size=1, seq_len=1024):")
    batch_size = 1
    sequence_length = 1024
    memory = calculate_memory_usage(params['total'], batch_size, sequence_length, d_model, num_layers)
    
    print(f"Parameters: {memory['parameters_gb']:.2f} GB")
    print(f"Activations: {memory['activations_gb']:.2f} GB")
    print(f"Gradients: {memory['gradients_gb']:.2f} GB") 
    print(f"AdamW optimizer state: {memory['optimizer_gb']:.2f} GB")
    print(f"TOTAL TRAINING MEMORY: {memory['total_gb']:.2f} GB")
    print()


def analyze_flops() -> None:
    """
    Parts (b) and (c): FLOP analysis for GPT-2 XL.
    """
    print("=== FLOP Analysis ===\n")
    
    # GPT-2 XL configuration
    vocab_size = 50257
    d_model = 1600
    num_layers = 48
    num_heads = 25
    d_ff = 6400
    
    # Example forward pass
    batch_size = 1
    sequence_length = 1024
    
    print("Part (b): Matrix Multiplications and FLOPs")
    print(f"Forward pass configuration: batch_size={batch_size}, seq_len={sequence_length}")
    print()
    
    flops = calculate_flops_per_forward_pass(batch_size, sequence_length, d_model, 
                                           num_layers, num_heads, d_ff, vocab_size)
    
    print("FLOPs per forward pass:")
    print(f"Token embeddings: {flops['token_embeddings']:,} FLOPs")
    print(f"Per layer:")
    print(f"  - QKV projections: {flops['qkv_projections_per_layer']:,} FLOPs")
    print(f"  - Attention scores (Q@K^T): {flops['attention_scores_per_layer']:,} FLOPs")
    print(f"  - Attention output (scores@V): {flops['attention_output_per_layer']:,} FLOPs")
    print(f"  - Output projection: {flops['output_projection_per_layer']:,} FLOPs")
    print(f"  - Attention total: {flops['attention_total_per_layer']:,} FLOPs")
    print(f"  - FFN W1: {flops['ffn_w1_per_layer']:,} FLOPs")
    print(f"  - FFN W3: {flops['ffn_w3_per_layer']:,} FLOPs") 
    print(f"  - FFN W2: {flops['ffn_w2_per_layer']:,} FLOPs")
    print(f"  - FFN total: {flops['ffn_total_per_layer']:,} FLOPs")
    print(f"  - Total per layer: {flops['total_per_layer']:,} FLOPs")
    print(f"All {num_layers} layers: {flops['all_layers']:,} FLOPs")
    print(f"LM head: {flops['lm_head']:,} FLOPs")
    print(f"TOTAL FLOPs: {flops['total']:,} FLOPs ({flops['total']/1e12:.2f}T)")
    print()
    
    # Part (c): Analysis of FLOP distribution
    print("Part (c): FLOP Distribution Analysis")
    total = flops['total']
    attention_total = flops['attention_total_per_layer'] * num_layers
    ffn_total = flops['ffn_total_per_layer'] * num_layers
    lm_head_total = flops['lm_head']
    
    print(f"Attention (all layers): {attention_total:,} FLOPs ({100*attention_total/total:.1f}%)")
    print(f"Feed-forward (all layers): {ffn_total:,} FLOPs ({100*ffn_total/total:.1f}%)")
    print(f"LM head: {lm_head_total:,} FLOPs ({100*lm_head_total/total:.1f}%)")
    print()
    
    # Within attention breakdown
    print("Within attention breakdown:")
    qkv_total = flops['qkv_projections_per_layer'] * num_layers
    scores_total = flops['attention_scores_per_layer'] * num_layers
    output_attn_total = flops['attention_output_per_layer'] * num_layers
    out_proj_total = flops['output_projection_per_layer'] * num_layers
    
    print(f"  QKV projections: {qkv_total:,} FLOPs ({100*qkv_total/attention_total:.1f}% of attention)")
    print(f"  Attention scores: {scores_total:,} FLOPs ({100*scores_total/attention_total:.1f}% of attention)")
    print(f"  Attention output: {output_attn_total:,} FLOPs ({100*output_attn_total/attention_total:.1f}% of attention)")
    print(f"  Output projection: {out_proj_total:,} FLOPs ({100*out_proj_total/attention_total:.1f}% of attention)")
    print()


def compare_gpt2_models() -> None:
    """
    Part (d): Compare resource usage across GPT-2 model sizes.
    """
    print("=== GPT-2 Model Size Comparison ===\n")
    
    # GPT-2 model configurations
    models = {
        'GPT-2 Small': {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072
        },
        'GPT-2 Medium': {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'd_ff': 4096
        },
        'GPT-2 Large': {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 1280,
            'num_layers': 36,
            'num_heads': 20,
            'd_ff': 5120
        },
        'GPT-2 XL': {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 1600,
            'num_layers': 48,
            'num_heads': 25,
            'd_ff': 6400
        }
    }
    
    print("Part (d): Parameter counts across model sizes")
    for name, config in models.items():
        params = count_parameters(**config)
        print(f"{name}: {params['total']:,} parameters ({params['total']/1e6:.1f}M)")
    
    print()
    print("FLOP counts and breakdown for single forward pass (batch_size=1, seq_len=1024):")
    print()
    
    # Store results for analysis
    results = {}
    
    for name, config in models.items():
        flops = calculate_flops_per_forward_pass(
            batch_size=1, 
            sequence_length=1024,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            vocab_size=config['vocab_size']
        )
        
        # Calculate component totals
        attention_total = flops['attention_total_per_layer'] * config['num_layers']
        ffn_total = flops['ffn_total_per_layer'] * config['num_layers']
        lm_head_total = flops['lm_head']
        total = flops['total']
        
        # Store for later analysis
        results[name] = {
            'attention': attention_total,
            'ffn': ffn_total,
            'lm_head': lm_head_total,
            'total': total
        }
        
        # Print breakdown for each model
        print(f"{name}:")
        print(f"  Total FLOPs: {total:,} ({total/1e12:.2f}T)")
        print(f"  - Attention: {attention_total:,} ({100*attention_total/total:.1f}%)")
        print(f"  - Feed-forward: {ffn_total:,} ({100*ffn_total/total:.1f}%)")
        print(f"  - LM head: {lm_head_total:,} ({100*lm_head_total/total:.1f}%)")
        print()
    
    # Analysis of proportional changes
    print("Analysis: How component proportions change with model size")
    print("=" * 60)
    print("As model size increases from Small to XL:")
    print()
    
    # Compare Small to XL
    small_attn_pct = 100 * results['GPT-2 Small']['attention'] / results['GPT-2 Small']['total']
    xl_attn_pct = 100 * results['GPT-2 XL']['attention'] / results['GPT-2 XL']['total']
    
    small_ffn_pct = 100 * results['GPT-2 Small']['ffn'] / results['GPT-2 Small']['total']
    xl_ffn_pct = 100 * results['GPT-2 XL']['ffn'] / results['GPT-2 XL']['total']
    
    small_lm_pct = 100 * results['GPT-2 Small']['lm_head'] / results['GPT-2 Small']['total']
    xl_lm_pct = 100 * results['GPT-2 XL']['lm_head'] / results['GPT-2 XL']['total']
    
    print(f"Attention: {small_attn_pct:.1f}% → {xl_attn_pct:.1f}% (slight decrease)")
    print(f"Feed-forward: {small_ffn_pct:.1f}% → {xl_ffn_pct:.1f}% (slight increase)")
    print(f"LM head: {small_lm_pct:.1f}% → {xl_lm_pct:.1f}% (significant decrease)")
    print()
    print("Key insight: As models scale, the LM head becomes a smaller proportion")
    print("of total FLOPs, while feed-forward networks slightly increase their")
    print("dominance. This is because the LM head scales linearly with d_model,")
    print("while transformer layers scale quadratically.")
    
    print()


def analyze_context_length_scaling() -> None:
    """
    Part (e): Analyze impact of increasing context length to 16,384.
    """
    print("=== Context Length Scaling Analysis ===\n")
    
    # GPT-2 XL configuration
    config = {
        'vocab_size': 50257,
        'd_model': 1600,
        'num_layers': 48,
        'num_heads': 25,
        'd_ff': 6400
    }
    
    context_lengths = [1024, 16384]
    batch_size = 1
    
    print("Part (e): Impact of increasing context length from 1,024 to 16,384")
    print()
    
    # Store results for comparison
    results = {}
    
    for context_length in context_lengths:
        print(f"Context length: {context_length:,}")
        
        flops = calculate_flops_per_forward_pass(
            batch_size=batch_size,
            sequence_length=context_length,
            **config
        )
        
        # Calculate totals for each component
        qkv_total = flops['qkv_projections_per_layer'] * config['num_layers']
        scores_total = flops['attention_scores_per_layer'] * config['num_layers']
        output_attn_total = flops['attention_output_per_layer'] * config['num_layers']
        out_proj_total = flops['output_projection_per_layer'] * config['num_layers']
        ffn_total = flops['ffn_total_per_layer'] * config['num_layers']
        lm_head_total = flops['lm_head']
        attention_total = qkv_total + scores_total + output_attn_total + out_proj_total
        
        results[context_length] = {
            'total': flops['total'],
            'qkv_projections': qkv_total,
            'attention_scores': scores_total,
            'attention_output': output_attn_total,
            'output_projection': out_proj_total,
            'attention_total': attention_total,
            'ffn_total': ffn_total,
            'lm_head': lm_head_total
        }
        
        print(f"  Total FLOPs: {flops['total']:,} ({flops['total']/1e12:.2f}T)")
        print(f"  Component breakdown:")
        print(f"    - QKV projections: {qkv_total:,} ({qkv_total/1e12:.2f}T, {100*qkv_total/flops['total']:.1f}%)")
        print(f"    - Attention scores (Q@K^T): {scores_total:,} ({scores_total/1e12:.2f}T, {100*scores_total/flops['total']:.1f}%)")
        print(f"    - Attention output (scores@V): {output_attn_total:,} ({output_attn_total/1e12:.2f}T, {100*output_attn_total/flops['total']:.1f}%)")
        print(f"    - Output projection: {out_proj_total:,} ({out_proj_total/1e12:.2f}T, {100*out_proj_total/flops['total']:.1f}%)")
        print(f"    - Feed-forward networks: {ffn_total:,} ({ffn_total/1e12:.2f}T, {100*ffn_total/flops['total']:.1f}%)")
        print(f"    - LM head: {lm_head_total:,} ({lm_head_total/1e12:.2f}T, {100*lm_head_total/flops['total']:.1f}%)")
        print()
    
    # Analyze scaling behavior
    print("Scaling Analysis:")
    print("=" * 50)
    scaling_factor = 16384 / 1024
    print(f"Context length increase: {scaling_factor:.1f}x (1,024 → 16,384)")
    print()
    
    print("Component scaling factors:")
    short_len, long_len = 1024, 16384
    
    components = [
        ("QKV projections", "qkv_projections", "Linear - scales with seq_len"),
        ("Attention scores", "attention_scores", "Quadratic - scales with seq_len²"),
        ("Attention output", "attention_output", "Quadratic - scales with seq_len²"), 
        ("Output projection", "output_projection", "Linear - scales with seq_len"),
        ("Feed-forward networks", "ffn_total", "Linear - scales with seq_len"),
        ("LM head", "lm_head", "Linear - scales with seq_len")
    ]
    
    for name, key, explanation in components:
        actual_scaling = results[long_len][key] / results[short_len][key]
        print(f"  {name}: {actual_scaling:.1f}x ({explanation})")
    
    total_scaling = results[long_len]['total'] / results[short_len]['total']
    print(f"  Total FLOPs: {total_scaling:.1f}x")
    print()
    
    # Analyze proportion changes
    print("How component proportions change:")
    print("-" * 40)
    for name, key, _ in components:
        short_pct = 100 * results[short_len][key] / results[short_len]['total']
        long_pct = 100 * results[long_len][key] / results[long_len]['total']
        change = long_pct - short_pct
        direction = "increases" if change > 0 else "decreases"
        print(f"  {name}: {short_pct:.1f}% → {long_pct:.1f}% ({direction} by {abs(change):.1f} percentage points)")
    
    print()
    print("Key insight: Quadratic attention components (scores and output) dominate")
    print(f"at longer sequences, growing from {100*(results[short_len]['attention_scores'] + results[short_len]['attention_output'])/results[short_len]['total']:.1f}% to {100*(results[long_len]['attention_scores'] + results[long_len]['attention_output'])/results[long_len]['total']:.1f}% of total FLOPs.")
    print("This makes long-context inference computationally expensive.")


if __name__ == "__main__":
    # Run all analyses
    gpt2_xl_analysis()
    analyze_flops()
    compare_gpt2_models()
    analyze_context_length_scaling()