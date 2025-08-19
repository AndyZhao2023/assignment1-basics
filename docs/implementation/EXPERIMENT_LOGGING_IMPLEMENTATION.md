# CS336 Assignment 1: Experiment Logging (3 Points) Implementation

This document describes the comprehensive implementation of the "Experiment logging (3 points)" problem from the CS336 assignment, which provides professional-grade experiment tracking and metrics logging for machine learning experiments.

## Overview

The experiment logging implementation provides a unified interface for tracking training metrics, hyperparameters, model architectures, and artifacts across multiple logging backends including Weights & Biases (wandb), local file-based logging, and console output.

## Architecture

### Core Components

1. **`ExperimentLogger` (Abstract Base Class)** - Defines the unified logging interface
2. **`WandbLogger`** - Weights & Biases cloud-based logging
3. **`ConsoleLogger`** - Local file and console logging
4. **`MultiLogger`** - Multi-backend logging support
5. **Logger Factory** - Convenient logger creation

## Implementation Details

### 1. ExperimentLogger Base Class

**File**: `cs336_basics/logger.py` - `ExperimentLogger`

The abstract base class defines a unified interface for all logging backends:

```python
class ExperimentLogger(ABC):
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None: ...
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None: ...
    @abstractmethod
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None: ...
    @abstractmethod
    def log_model(self, model: torch.nn.Module, name: str = "model") -> None: ...
    @abstractmethod
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "file") -> None: ...
    @abstractmethod
    def finish(self) -> None: ...
```

**Built-in Utility Methods**:
- `log_gradient_norms()` - Automatically compute and log gradient norms
- `log_learning_rate()` - Extract and log learning rates from optimizer
- `log_system_metrics()` - Log GPU memory and system metrics

### 2. Weights & Biases Logger

**File**: `cs336_basics/logger.py` - `WandbLogger`

Provides cloud-based experiment tracking with rich visualization:

**Features**:
- ✅ Automatic project and run management
- ✅ Real-time metric streaming to W&B dashboard
- ✅ Hyperparameter tracking and comparison
- ✅ Model architecture visualization
- ✅ Artifact storage and versioning
- ✅ Histogram logging for weights and gradients
- ✅ System monitoring integration
- ✅ Collaborative experiment sharing

**Usage**:
```python
logger = WandbLogger(
    project_name="cs336-training",
    experiment_name="transformer_experiment",
    entity="your_username",
    tags=["transformer", "language-model"],
    mode="online"  # or "offline" for local logging
)
```

### 3. Console and File Logger

**File**: `cs336_basics/logger.py` - `ConsoleLogger`

Provides local logging with JSON, CSV, and console output:

**Features**:
- ✅ JSON-based metrics storage
- ✅ CSV export for easy analysis
- ✅ Real-time console output with configurable intervals
- ✅ Automatic directory structure creation
- ✅ Metadata tracking (timestamps, experiment info)
- ✅ Artifact copying and organization
- ✅ Batch writing for performance
- ✅ Histogram statistics logging

**Output Structure**:
```
logs/
├── project_name/
│   └── experiment_name/
│       ├── config.json              # Hyperparameters
│       ├── metrics.json             # All metrics (timestamped)
│       ├── metrics.csv              # CSV export for analysis
│       ├── metadata.json            # Experiment metadata
│       ├── model_architecture.txt   # Model description
│       └── artifacts/               # Saved artifacts
│           └── checkpoints/
```

### 4. Multi-Backend Logger

**File**: `cs336_basics/logger.py` - `MultiLogger`

Enables simultaneous logging to multiple backends:

**Features**:
- ✅ Logs to multiple services simultaneously
- ✅ Transparent interface (same as single logger)
- ✅ Automatic error handling per backend
- ✅ Configurable backend combinations

**Usage**:
```python
# Automatic multi-backend setup
logger = create_logger(
    backend="multi",
    project_name="cs336-training",
    experiment_name="multi_backend_experiment"
)
```

### 5. Enhanced Training Integration

**File**: `train_with_logging.py`

Comprehensive training script with full logging integration:

**Logged Metrics**:
- **Training Metrics**: Loss, perplexity, accuracy
- **Optimization**: Learning rate schedules, gradient norms (before/after clipping)
- **Performance**: Tokens/second, iterations/second, timing breakdown
- **System**: GPU memory usage, elapsed time
- **Model**: Weight histograms, gradient histograms, parameter counts

**Command Line Interface**:
```bash
# W&B logging
uv run python train_with_logging.py \
  --use_wandb \
  --wandb_project "cs336-experiments" \
  --experiment_name "transformer_large" \
  --log_histograms

# Console logging only
uv run python train_with_logging.py \
  --log_dir "my_experiments" \
  --log_interval 5
```

## Key Features

### Comprehensive Metrics Tracking

**Training Metrics**:
```python
metrics = {
    "train/loss": loss.item(),
    "train/perplexity": np.exp(loss.item()),
    "train/grad_norm_before_clip": total_norm_before,
    "train/grad_norm_after_clip": total_norm_after,
    "train/grad_clip_ratio": total_norm_after / (total_norm_before + 1e-8),
    "timing/forward_ms": forward_time * 1000,
    "timing/backward_ms": backward_time * 1000,
    "performance/tokens_per_second": tokens_per_sec
}
```

**System Monitoring**:
```python
# Automatic GPU memory tracking
metrics = {
    "system/gpu_0_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
    "system/gpu_0_memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
    "system/elapsed_time_hours": (time.time() - start_time) / 3600
}
```

### Advanced Logging Features

**Gradient Norm Tracking**:
```python
# Automatic gradient norm computation and logging
grad_norms = logger.log_gradient_norms(model, step)
# Logs: grad_norm/layer1.weight, grad_norm/layer2.bias, grad_norm/total
```

**Learning Rate Monitoring**:
```python
# Automatic learning rate extraction from optimizer
logger.log_learning_rate(optimizer, step)
# Logs: learning_rate/group_0, learning_rate/group_1, etc.
```

**Histogram Logging**:
```python
# Weight and gradient distributions
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.log_histogram(f"weights/{name}", param.data, step)
        if param.grad is not None:
            logger.log_histogram(f"gradients/{name}", param.grad, step)
```

### Artifact Management

**Automatic Checkpoint Tracking**:
```python
# Save and log checkpoints
save_checkpoint(model, optimizer, iteration, checkpoint_path)
logger.log_artifact(str(checkpoint_path), f"checkpoint_{iteration:06d}.pt", "checkpoint")
```

**Configuration Versioning**:
```python
# Automatically log training configuration
logger.log_artifact(str(config_path), "training_config.json", "config")
```

## Usage Examples

### Basic Usage

```python
from cs336_basics.logger import create_logger

# Create logger
logger = create_logger(
    backend="wandb",  # or "console" or "multi"
    project_name="cs336-experiments",
    experiment_name="my_experiment"
)

# Log hyperparameters
logger.log_hyperparameters({
    "learning_rate": 3e-4,
    "batch_size": 32,
    "model_size": "small"
})

# Training loop
for step in range(1000):
    # ... training code ...
    
    # Log metrics
    logger.log_metrics({
        "train/loss": loss.item(),
        "train/accuracy": accuracy,
        "learning_rate": current_lr
    }, step)
    
    # Log gradients periodically
    if step % 100 == 0:
        logger.log_gradient_norms(model, step)

# Finish logging
logger.finish()
```

### Advanced Integration

```python
# Multi-backend logging with comprehensive tracking
logger = create_logger(
    backend="multi",
    project_name="cs336-research",
    experiment_name="ablation_study_1",
    wandb_entity="research_team",
    tags=["ablation", "transformer", "efficiency"]
)

# Log model architecture
logger.log_model(model, "transformer_v2")

# Training with rich logging
for iteration in range(max_iters):
    # Training step
    metrics = train_step(model, batch, optimizer, logger=logger, step=iteration)
    
    # Additional system monitoring
    logger.log_system_metrics(iteration)
    
    # Periodic evaluation
    if iteration % eval_interval == 0:
        eval_metrics = evaluate_model(model, eval_data, logger=logger, step=iteration)
    
    # Checkpoint saving with artifact tracking
    if iteration % save_interval == 0:
        checkpoint_path = save_checkpoint(model, optimizer, iteration)
        logger.log_artifact(checkpoint_path, f"checkpoint_{iteration}", "checkpoint")

logger.finish()
```

### Research Workflow

```python
# Hyperparameter sweep with logging
import itertools

hyperparams = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "num_layers": [4, 6, 8]
}

for i, (lr, bs, layers) in enumerate(itertools.product(*hyperparams.values())):
    logger = create_logger(
        backend="wandb",
        project_name="hyperparameter_sweep",
        experiment_name=f"sweep_run_{i:03d}",
        tags=["sweep", f"lr_{lr}", f"bs_{bs}", f"layers_{layers}"]
    )
    
    config = {
        "learning_rate": lr,
        "batch_size": bs,
        "num_layers": layers,
        "sweep_id": i
    }
    logger.log_hyperparameters(config)
    
    # Run training with this configuration
    final_loss = train_model(config, logger)
    
    logger.log_metrics({"final/loss": final_loss}, step=0)
    logger.finish()
```

## Performance Considerations

### Efficient Logging

**Batch Writing**: Console logger batches metrics to reduce I/O:
```python
# Writes metrics in batches of 10 for efficiency
self.metrics_buffer.append(metrics_with_meta)
if len(self.metrics_buffer) >= 10:
    self._flush_metrics()
```

**Configurable Intervals**: Control logging frequency to balance detail vs. performance:
```python
# Log detailed metrics every N steps
if iteration % args.log_interval == 0:
    logger.log_metrics(detailed_metrics, iteration)
    logger.log_system_metrics(iteration)
```

**Histogram Optimization**: Log histograms less frequently for large models:
```python
# Log histograms every 10x the normal interval
if args.log_histograms and iteration % (args.log_interval * 10) == 0:
    for name, param in model.named_parameters():
        logger.log_histogram(f"weights/{name}", param.data, iteration)
```

### Memory Management

**Tensor Detachment**: Automatically detach tensors before logging:
```python
if isinstance(values, torch.Tensor):
    values = values.detach().cpu().numpy()
```

**Gradient Cleanup**: Safe gradient norm computation:
```python
total_norm = 0.0
for param in model.parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2).item()
        total_norm += param_norm ** 2
total_norm = total_norm ** 0.5
```

## Testing and Validation

### Test Coverage

**File**: `test_experiment_logging.py`

Comprehensive test suite covering:
- ✅ Console logger functionality (metrics, artifacts, histograms)
- ✅ W&B logger integration (mocked for CI/CD)
- ✅ Multi-logger coordination
- ✅ Factory function behavior
- ✅ End-to-end integration testing
- ✅ Error handling and edge cases

**Running Tests**:
```bash
# Run full test suite
uv run python test_experiment_logging.py

# Or with pytest if available
uv run pytest test_experiment_logging.py -v
```

### Demonstration

**File**: `demo_experiment_logging.py`

Interactive demonstration showing:
- Console logging with file output
- W&B integration (when available)
- Multi-backend coordination
- Factory usage patterns
- Real metrics tracking examples

**Running Demo**:
```bash
uv run python demo_experiment_logging.py
```

## Integration with Existing Training

The logging system integrates seamlessly with the existing training infrastructure:

### With Basic Training Script
```bash
# Enhanced training with logging
uv run python train_with_logging.py \
  --use_wandb \
  --wandb_project "cs336-experiments" \
  --max_iters 1000 \
  --log_interval 10 \
  --log_histograms
```

### With Checkpointing
```python
# Automatic checkpoint artifact tracking
checkpoint_path = save_checkpoint(model, optimizer, iteration)
logger.log_artifact(checkpoint_path, f"checkpoint_{iteration}", "checkpoint")

# Resume with logging continuity
if args.resume_from:
    iteration = load_checkpoint(args.resume_from, model, optimizer)
    logger.log_metrics({"training/resumed_from_iteration": iteration}, iteration)
```

### With Evaluation
```python
# Comprehensive evaluation logging
def evaluate_model(model, dataset, logger=None, step=None):
    # ... evaluation code ...
    
    metrics = {
        "eval/loss": avg_loss,
        "eval/perplexity": np.exp(avg_loss),
        "eval/num_batches": num_batches,
        "timing/eval_total_seconds": eval_time
    }
    
    if logger:
        logger.log_metrics(metrics, step)
    
    return metrics
```

## Configuration and Customization

### Environment Setup

**Weights & Biases Setup**:
```bash
# Install and login to W&B
pip install wandb
wandb login

# Or set API key
export WANDB_API_KEY="your_api_key"
```

**Project Configuration**:
```python
# Custom W&B configuration
logger = WandbLogger(
    project_name="cs336-research",
    entity="your_team",
    tags=["baseline", "transformer"],
    mode="online",  # "offline" for local logging
    config=initial_config
)
```

### Custom Backends

The architecture supports easy extension with custom backends:

```python
class CustomLogger(ExperimentLogger):
    def log_metrics(self, metrics, step=None):
        # Custom implementation (e.g., database, cloud service)
        pass
    
    # Implement other abstract methods...
```

## Best Practices

### Experiment Organization

1. **Consistent Naming**: Use descriptive project and experiment names
2. **Meaningful Tags**: Tag experiments for easy filtering and grouping
3. **Hyperparameter Logging**: Always log complete configuration
4. **Artifact Tracking**: Version control checkpoints and important files

### Performance Optimization

1. **Logging Intervals**: Balance detail with performance impact
2. **Histogram Frequency**: Log histograms less frequently for large models
3. **Batch Processing**: Use console logger's batch writing for efficiency
4. **System Monitoring**: Monitor resource usage to optimize training

### Reproducibility

1. **Seed Logging**: Always log random seeds
2. **Environment Info**: Log system and library versions
3. **Configuration Files**: Version control all configuration
4. **Checkpoint Strategy**: Save regular checkpoints with artifacts

## Conclusion

The experiment logging implementation provides a production-ready solution for tracking machine learning experiments with:

✅ **Multiple Backend Support**: W&B, console, and multi-backend logging  
✅ **Comprehensive Metrics**: Training, system, and performance monitoring  
✅ **Rich Visualization**: Histograms, learning curves, and system metrics  
✅ **Artifact Management**: Automatic checkpoint and configuration tracking  
✅ **Integration Ready**: Seamless integration with existing training pipelines  
✅ **Performance Optimized**: Efficient logging with configurable intervals  
✅ **Well Tested**: Comprehensive test coverage and validation  
✅ **Production Ready**: Error handling, documentation, and best practices

This implementation successfully addresses all requirements of the "Experiment logging (3 points)" problem and provides a professional-grade foundation for machine learning experiment management and reproducibility.

## Quick Start

Get started with experiment logging in 3 steps:

1. **Install dependencies** (W&B is optional):
   ```bash
   pip install wandb  # Optional for cloud logging
   ```

2. **Basic usage**:
   ```python
   from cs336_basics.logger import create_logger
   
   logger = create_logger("console", "my_project", "experiment_1")
   logger.log_hyperparameters({"lr": 0.001, "batch_size": 32})
   logger.log_metrics({"loss": 2.5}, step=1)
   logger.finish()
   ```

3. **Enhanced training**:
   ```bash
   uv run python train_with_logging.py --use_wandb --wandb_project "cs336"
   ```

The implementation is ready for immediate use and provides comprehensive experiment tracking capabilities for the CS336 assignment and beyond.