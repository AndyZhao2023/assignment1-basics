# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CS336 Spring 2025 Assignment 1: Basics - a Stanford course assignment implementing fundamental components of transformer models and NLP tools.

## Development Commands

### Environment Setup
- Uses `uv` for dependency management
- Run any Python file: `uv run <python_file_path>`

### Testing
- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest tests/test_model.py::test_linear`
- Run tests matching pattern: `uv run pytest -k "test_name_pattern"`

### Linting
- Check code style: `uv run ruff check .`
- Fix code style issues: `uv run ruff check . --fix`

## Code Architecture

### Core Implementation Structure

The main implementation is split across:
- `cs336_basics/nn.py`: Neural network components (attention, layers, activations)
- `cs336_basics/tokenizer.py`: BPE tokenizer implementation
- `tests/adapters.py`: Bridge functions connecting implementations to tests

### Key Components to Implement

1. **Neural Network Components** (`cs336_basics/nn.py`):
   - RMSNorm layer with float32 upcasting
   - Rotary Position Embedding (RoPE)
   - SwiGLU feedforward network
   - Scaled dot-product attention
   - Multi-head self-attention (with and without RoPE)
   - Transformer blocks and full language model

2. **Tokenizer** (`cs336_basics/tokenizer.py`):
   - BPE tokenizer with special token support
   - Training algorithm for BPE

3. **Training Utilities**:
   - Cross-entropy loss
   - Gradient clipping
   - AdamW optimizer
   - Cosine learning rate schedule
   - Checkpointing (save/load)
   - Data loading (get_batch)

### Test Adapters Pattern

The `tests/adapters.py` file contains wrapper functions that connect your implementations to the test suite. Each `run_*` function:
- Takes test parameters and weights
- Instantiates your implementation
- Sets the provided weights
- Returns the output

Already implemented adapters:
- `run_linear`, `run_embedding`, `run_swiglu`
- `run_scaled_dot_product_attention`
- `run_rope`, `run_rmsnorm`, `run_silu`, `run_softmax`
- `get_tokenizer`, `run_train_bpe`

Still need implementation:
- `run_multihead_self_attention`, `run_multihead_self_attention_with_rope`
- `run_transformer_block`, `run_transformer_lm`
- `run_get_batch`, `run_cross_entropy`
- `run_gradient_clipping`, `get_adamw_cls`
- `run_get_lr_cosine_schedule`
- `run_save_checkpoint`, `run_load_checkpoint`

## Testing Notes

- Tests use snapshot testing with reference implementations in `tests/_snapshots/`
- Tests verify numerical accuracy and compatibility with PyTorch implementations
- Use `pytest -s` to see print outputs during testing (already configured in pyproject.toml)