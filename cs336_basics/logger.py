"""
Experiment logging utilities for CS336 Assignment 1.
Implements comprehensive logging with support for Weights & Biases, TensorBoard, and console output.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import csv

import torch
import numpy as np


class ExperimentLogger(ABC):
    """
    Abstract base class for experiment logging.
    
    Provides a unified interface for logging metrics, hyperparameters, and artifacts
    across different logging backends (wandb, tensorboard, console).
    """
    
    def __init__(self, project_name: str, experiment_name: Optional[str] = None):
        """
        Initialize the experiment logger.
        
        Args:
            project_name: Name of the project/experiment group
            experiment_name: Specific name for this experiment run
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.start_time = time.time()
        self.step = 0
        
    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for the experiment."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log scalar metrics."""
        pass
    
    @abstractmethod
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None:
        """Log histogram of values (e.g., weights, gradients)."""
        pass
    
    @abstractmethod
    def log_model(self, model: torch.nn.Module, name: str = "model") -> None:
        """Log model architecture and optionally weights."""
        pass
    
    @abstractmethod
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "file") -> None:
        """Log a file artifact (e.g., checkpoint, config)."""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finalize logging and cleanup resources."""
        pass
    
    def log_gradient_norms(self, model: torch.nn.Module, step: Optional[int] = None) -> Dict[str, float]:
        """
        Log gradient norms for model parameters.
        
        Args:
            model: PyTorch model
            step: Current training step
            
        Returns:
            Dictionary of gradient norms
        """
        grad_norms = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[f"grad_norm/{name}"] = param_norm
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        grad_norms["grad_norm/total"] = total_norm
        
        self.log_metrics(grad_norms, step)
        return grad_norms
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None) -> None:
        """Log current learning rate from optimizer."""
        lrs = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lrs[f"learning_rate/group_{i}"] = param_group['lr']
        self.log_metrics(lrs, step)
    
    def log_system_metrics(self, step: Optional[int] = None) -> None:
        """Log system metrics like GPU memory usage."""
        metrics = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"system/gpu_{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                metrics[f"system/gpu_{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
                
        # Log elapsed time
        metrics["system/elapsed_time_hours"] = (time.time() - self.start_time) / 3600
        
        self.log_metrics(metrics, step)


class WandbLogger(ExperimentLogger):
    """
    Weights & Biases logger implementation.
    
    Provides cloud-based experiment tracking with rich visualization
    and collaboration features.
    """
    
    def __init__(
        self, 
        project_name: str, 
        experiment_name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        reinit: bool = True,
        mode: str = "online"
    ):
        """
        Initialize Weights & Biases logger.
        
        Args:
            project_name: W&B project name
            experiment_name: Run name
            entity: W&B entity (username or team)
            tags: List of tags for the run
            config: Initial configuration dictionary
            reinit: Whether to reinitialize if a run exists
            mode: "online", "offline", or "disabled"
        """
        super().__init__(project_name, experiment_name)
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        # Initialize W&B run
        self.run = self.wandb.init(
            project=project_name,
            name=experiment_name,
            entity=entity,
            tags=tags or [],
            config=config or {},
            reinit=reinit,
            mode=mode
        )
        
        self.artifact_count = 0
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B config."""
        self.wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics, step=self.step)
            self.step += 1
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None:
        """Log histogram to W&B."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        histogram = self.wandb.Histogram(values)
        self.log_metrics({name: histogram}, step)
    
    def log_model(self, model: torch.nn.Module, name: str = "model") -> None:
        """Log model architecture to W&B."""
        # Log model summary
        self.wandb.watch(model, log="all", log_freq=100)
        
        # Log model graph if possible
        try:
            dummy_input = torch.randn(1, 128)  # Adjust based on your model
            self.wandb.log({f"{name}_graph": self.wandb.Graph(model, dummy_input)})
        except:
            pass  # Graph logging might fail for complex models
    
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "file") -> None:
        """Log artifact to W&B."""
        artifact_name = name or f"artifact_{self.artifact_count}"
        artifact = self.wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)
        self.artifact_count += 1
    
    def finish(self) -> None:
        """Finish W&B run."""
        self.wandb.finish()


class ConsoleLogger(ExperimentLogger):
    """
    Console and file-based logger implementation.
    
    Provides local logging with JSON and CSV output for experiments
    without cloud connectivity.
    """
    
    def __init__(
        self, 
        project_name: str, 
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        print_interval: int = 10
    ):
        """
        Initialize console logger.
        
        Args:
            project_name: Project name
            experiment_name: Experiment name
            log_dir: Directory for log files
            print_interval: How often to print to console
        """
        super().__init__(project_name, experiment_name)
        
        # Setup logging directory
        self.log_dir = Path(log_dir or f"logs/{project_name}/{self.experiment_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.print_interval = print_interval
        
        # Initialize log files
        self.metrics_file = self.log_dir / "metrics.json"
        self.csv_file = self.log_dir / "metrics.csv"
        self.config_file = self.log_dir / "config.json"
        
        self.metrics_buffer = []
        self.csv_writer = None
        self.csv_file_handle = None
        
        # Write initial metadata
        metadata = {
            "project": project_name,
            "experiment": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": self.start_time
        }
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters to config file."""
        with open(self.config_file, "w") as f:
            json.dump(params, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to console and file."""
        current_step = step if step is not None else self.step
        
        # Add timestamp and step to metrics
        metrics_with_meta = {
            "step": current_step,
            "timestamp": time.time(),
            **metrics
        }
        
        # Append to JSON file
        self.metrics_buffer.append(metrics_with_meta)
        if len(self.metrics_buffer) >= 10:  # Write in batches
            self._flush_metrics()
        
        # Write to CSV
        self._write_csv(metrics_with_meta)
        
        # Print to console at intervals
        if current_step % self.print_interval == 0:
            print(f"Step {current_step:6d} | ", end="")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f} | ", end="")
                else:
                    print(f"{key}: {value} | ", end="")
            print()
        
        if step is None:
            self.step += 1
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None:
        """Log histogram statistics to console."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        stats = {
            f"{name}/mean": float(np.mean(values)),
            f"{name}/std": float(np.std(values)),
            f"{name}/min": float(np.min(values)),
            f"{name}/max": float(np.max(values)),
            f"{name}/median": float(np.median(values))
        }
        
        self.log_metrics(stats, step)
    
    def log_model(self, model: torch.nn.Module, name: str = "model") -> None:
        """Log model summary to file."""
        model_file = self.log_dir / f"{name}_architecture.txt"
        
        with open(model_file, "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(str(model))
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        
        print(f"Model architecture saved to {model_file}")
    
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "file") -> None:
        """Copy artifact to log directory."""
        from shutil import copy2
        
        artifact_dir = self.log_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        dest_name = name or Path(file_path).name
        dest_path = artifact_dir / dest_name
        
        copy2(file_path, dest_path)
        print(f"Artifact saved: {dest_path}")
    
    def _flush_metrics(self) -> None:
        """Flush metrics buffer to JSON file."""
        if not self.metrics_buffer:
            return
        
        # Append to existing metrics
        existing_metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                try:
                    existing_metrics = json.load(f)
                except:
                    existing_metrics = []
        
        existing_metrics.extend(self.metrics_buffer)
        
        with open(self.metrics_file, "w") as f:
            json.dump(existing_metrics, f, indent=2, default=str)
        
        self.metrics_buffer = []
    
    def _write_csv(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to CSV file."""
        if self.csv_writer is None:
            self.csv_file_handle = open(self.csv_file, "w", newline="")
            fieldnames = list(metrics.keys())
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
            self.csv_writer.writeheader()
        
        # Handle new fields
        if set(metrics.keys()) - set(self.csv_writer.fieldnames):
            # Recreate CSV with new fields
            self.csv_file_handle.close()
            
            # Read existing data
            existing_data = []
            if self.csv_file.exists():
                with open(self.csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
            
            # Write with new fields
            self.csv_file_handle = open(self.csv_file, "w", newline="")
            fieldnames = list(set(self.csv_writer.fieldnames) | set(metrics.keys()))
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            
            for row in existing_data:
                self.csv_writer.writerow(row)
        
        self.csv_writer.writerow(metrics)
        self.csv_file_handle.flush()
    
    def finish(self) -> None:
        """Finalize logging and close files."""
        # Flush remaining metrics
        self._flush_metrics()
        
        # Close CSV file
        if self.csv_file_handle:
            self.csv_file_handle.close()
        
        # Write final metadata
        metadata = {
            "project": self.project_name,
            "experiment": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_hours": (time.time() - self.start_time) / 3600,
            "total_steps": self.step
        }
        with open(self.log_dir / "metadata_final.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nLogging finished. Results saved to {self.log_dir}")


class MultiLogger(ExperimentLogger):
    """
    Composite logger that logs to multiple backends simultaneously.
    
    Useful for logging to both W&B and local files, or multiple services.
    """
    
    def __init__(self, loggers: list[ExperimentLogger]):
        """
        Initialize multi-logger.
        
        Args:
            loggers: List of logger instances to use
        """
        if not loggers:
            raise ValueError("At least one logger must be provided")
        
        # Use first logger's project and experiment names
        super().__init__(loggers[0].project_name, loggers[0].experiment_name)
        self.loggers = loggers
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all loggers."""
        for logger in self.loggers:
            logger.log_hyperparameters(params)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to all loggers."""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None:
        """Log histogram to all loggers."""
        for logger in self.loggers:
            logger.log_histogram(name, values, step)
    
    def log_model(self, model: torch.nn.Module, name: str = "model") -> None:
        """Log model to all loggers."""
        for logger in self.loggers:
            logger.log_model(model, name)
    
    def log_artifact(self, file_path: str, name: Optional[str] = None, artifact_type: str = "file") -> None:
        """Log artifact to all loggers."""
        for logger in self.loggers:
            logger.log_artifact(file_path, name, artifact_type)
    
    def finish(self) -> None:
        """Finish all loggers."""
        for logger in self.loggers:
            logger.finish()


def create_logger(
    backend: str = "console",
    project_name: str = "cs336_experiment",
    experiment_name: Optional[str] = None,
    **kwargs
) -> ExperimentLogger:
    """
    Factory function to create a logger instance.
    
    Args:
        backend: "wandb", "console", or "multi"
        project_name: Project name
        experiment_name: Experiment name
        **kwargs: Additional backend-specific arguments
        
    Returns:
        ExperimentLogger instance
    """
    if backend == "wandb":
        return WandbLogger(project_name, experiment_name, **kwargs)
    elif backend == "console":
        return ConsoleLogger(project_name, experiment_name, **kwargs)
    elif backend == "multi":
        # Create both wandb and console loggers
        loggers = [
            WandbLogger(project_name, experiment_name, **kwargs),
            ConsoleLogger(project_name, experiment_name, **kwargs)
        ]
        return MultiLogger(loggers)
    else:
        raise ValueError(f"Unknown backend: {backend}")