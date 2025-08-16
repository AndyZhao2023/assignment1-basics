#!/usr/bin/env python3
"""
Test suite for the experiment logging implementation.
Tests console logging, W&B integration, and multi-backend functionality.
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import csv

import torch
import numpy as np
import pytest

from cs336_basics.logger import (
    ConsoleLogger, WandbLogger, MultiLogger, create_logger, ExperimentLogger
)
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW


class TestConsoleLogger:
    """Test the console and file-based logger"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ConsoleLogger(
            project_name="test_project",
            experiment_name="test_experiment",
            log_dir=self.temp_dir,
            print_interval=1
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        self.logger.finish()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test logger initialization"""
        assert self.logger.project_name == "test_project"
        assert self.logger.experiment_name == "test_experiment"
        assert Path(self.temp_dir).exists()
        
        # Check that required files are created
        log_dir = Path(self.temp_dir) / "test_project" / "test_experiment"
        assert log_dir.exists()
        assert (log_dir / "metadata.json").exists()
    
    def test_hyperparameter_logging(self):
        """Test hyperparameter logging"""
        params = {
            "learning_rate": 3e-4,
            "batch_size": 32,
            "model_type": "transformer",
            "layers": 12
        }
        
        self.logger.log_hyperparameters(params)
        
        # Check config file was created
        config_file = Path(self.temp_dir) / "test_project" / "test_experiment" / "config.json"
        assert config_file.exists()
        
        # Verify contents
        with open(config_file, "r") as f:
            saved_params = json.load(f)
        
        assert saved_params == params
    
    def test_metrics_logging(self):
        """Test metrics logging to JSON and CSV"""
        metrics1 = {"loss": 2.5, "accuracy": 0.85, "step": 1}
        metrics2 = {"loss": 2.3, "accuracy": 0.87, "step": 2}
        
        self.logger.log_metrics(metrics1, step=1)
        self.logger.log_metrics(metrics2, step=2)
        
        # Force flush of metrics
        self.logger._flush_metrics()
        
        # Check JSON file
        metrics_file = Path(self.temp_dir) / "test_project" / "test_experiment" / "metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file, "r") as f:
            saved_metrics = json.load(f)
        
        assert len(saved_metrics) >= 2
        assert saved_metrics[0]["loss"] == 2.5
        assert saved_metrics[1]["loss"] == 2.3
        
        # Check CSV file
        csv_file = Path(self.temp_dir) / "test_project" / "test_experiment" / "metrics.csv"
        assert csv_file.exists()
        
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) >= 2
        assert float(rows[0]["loss"]) == 2.5
        assert float(rows[1]["loss"]) == 2.3
    
    def test_histogram_logging(self):
        """Test histogram logging (should log statistics)"""
        values = torch.randn(1000)
        self.logger.log_histogram("test_weights", values, step=1)
        
        # Should have logged statistics
        self.logger._flush_metrics()
        
        metrics_file = Path(self.temp_dir) / "test_project" / "test_experiment" / "metrics.json"
        with open(metrics_file, "r") as f:
            saved_metrics = json.load(f)
        
        # Check that histogram statistics were logged
        metric_names = set()
        for metric in saved_metrics:
            metric_names.update(metric.keys())
        
        assert "test_weights/mean" in metric_names
        assert "test_weights/std" in metric_names
        assert "test_weights/min" in metric_names
        assert "test_weights/max" in metric_names
    
    def test_model_logging(self):
        """Test model architecture logging"""
        model = TransformerLM(
            vocab_size=100,
            context_length=64,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=256
        )
        
        self.logger.log_model(model, "test_model")
        
        # Check that model file was created
        model_file = Path(self.temp_dir) / "test_project" / "test_experiment" / "test_model_architecture.txt"
        assert model_file.exists()
        
        # Verify contents include model info
        with open(model_file, "r") as f:
            content = f.read()
        
        assert "test_model" in content
        assert "Total parameters:" in content
        assert "Trainable parameters:" in content
    
    def test_artifact_logging(self):
        """Test artifact logging"""
        # Create a temporary file to log as artifact
        test_file = Path(self.temp_dir) / "test_artifact.txt"
        with open(test_file, "w") as f:
            f.write("This is a test artifact")
        
        self.logger.log_artifact(str(test_file), "test_artifact.txt", "test")
        
        # Check that artifact was copied
        artifact_dir = Path(self.temp_dir) / "test_project" / "test_experiment" / "artifacts"
        copied_file = artifact_dir / "test_artifact.txt"
        
        assert copied_file.exists()
        with open(copied_file, "r") as f:
            content = f.read()
        assert content == "This is a test artifact"
    
    def test_gradient_norm_logging(self):
        """Test gradient norm logging functionality"""
        model = TransformerLM(
            vocab_size=50,
            context_length=32,
            d_model=64,
            num_layers=1,
            num_heads=2,
            d_ff=128
        )
        
        # Create fake gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.01
        
        grad_norms = self.logger.log_gradient_norms(model, step=1)
        
        # Check that gradient norms were calculated
        assert "grad_norm/total" in grad_norms
        assert grad_norms["grad_norm/total"] > 0
        
        # Check that individual parameter norms were logged
        param_names = [name for name, _ in model.named_parameters()]
        for name in param_names:
            assert f"grad_norm/{name}" in grad_norms


class TestWandbLogger:
    """Test the Weights & Biases logger with mocking"""
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_initialization(self, mock_wandb):
        """Test W&B logger initialization"""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        logger = WandbLogger(
            project_name="test_project",
            experiment_name="test_experiment",
            entity="test_entity",
            tags=["test", "demo"]
        )
        
        # Verify wandb.init was called with correct arguments
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            name="test_experiment",
            entity="test_entity",
            tags=["test", "demo"],
            config={},
            reinit=True,
            mode="online"
        )
        
        assert logger.run == mock_run
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_hyperparameter_logging(self, mock_wandb):
        """Test W&B hyperparameter logging"""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        mock_config = Mock()
        mock_wandb.config = mock_config
        
        logger = WandbLogger("test_project", "test_experiment")
        
        params = {"lr": 3e-4, "batch_size": 32}
        logger.log_hyperparameters(params)
        
        mock_config.update.assert_called_once_with(params)
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_metrics_logging(self, mock_wandb):
        """Test W&B metrics logging"""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        logger = WandbLogger("test_project", "test_experiment")
        
        metrics = {"loss": 2.5, "accuracy": 0.85}
        logger.log_metrics(metrics, step=10)
        
        mock_wandb.log.assert_called_once_with(metrics, step=10)
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_histogram_logging(self, mock_wandb):
        """Test W&B histogram logging"""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        mock_histogram = Mock()
        mock_wandb.Histogram.return_value = mock_histogram
        
        logger = WandbLogger("test_project", "test_experiment")
        
        values = torch.randn(100)
        logger.log_histogram("test_hist", values, step=5)
        
        # Check that histogram was created and logged
        mock_wandb.Histogram.assert_called_once()
        mock_wandb.log.assert_called_once_with({"test_hist": mock_histogram}, step=5)
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_artifact_logging(self, mock_wandb):
        """Test W&B artifact logging"""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        mock_artifact = Mock()
        mock_wandb.Artifact.return_value = mock_artifact
        
        logger = WandbLogger("test_project", "test_experiment")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            logger.log_artifact(temp_file, "test_artifact", "checkpoint")
            
            # Verify artifact creation and logging
            mock_wandb.Artifact.assert_called_once_with("test_artifact", type="checkpoint")
            mock_artifact.add_file.assert_called_once_with(temp_file)
            mock_run.log_artifact.assert_called_once_with(mock_artifact)
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestMultiLogger:
    """Test the multi-backend logger"""
    
    def test_multi_logger_initialization(self):
        """Test multi-logger initialization"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            logger1 = ConsoleLogger("test_project", "test_exp", temp_dir)
            logger2 = ConsoleLogger("test_project", "test_exp", temp_dir)
            
            multi = MultiLogger([logger1, logger2])
            
            assert multi.project_name == "test_project"
            assert multi.experiment_name == "test_exp"
            assert len(multi.loggers) == 2
            
            multi.finish()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_multi_logger_propagation(self):
        """Test that multi-logger propagates calls to all loggers"""
        logger1 = Mock(spec=ExperimentLogger)
        logger1.project_name = "test"
        logger1.experiment_name = "test"
        
        logger2 = Mock(spec=ExperimentLogger)
        logger2.project_name = "test"
        logger2.experiment_name = "test"
        
        multi = MultiLogger([logger1, logger2])
        
        # Test hyperparameter logging
        params = {"lr": 0.001}
        multi.log_hyperparameters(params)
        
        logger1.log_hyperparameters.assert_called_once_with(params)
        logger2.log_hyperparameters.assert_called_once_with(params)
        
        # Test metrics logging
        metrics = {"loss": 1.5}
        multi.log_metrics(metrics, step=10)
        
        logger1.log_metrics.assert_called_once_with(metrics, 10)
        logger2.log_metrics.assert_called_once_with(metrics, 10)
        
        # Test finish
        multi.finish()
        logger1.finish.assert_called_once()
        logger2.finish.assert_called_once()


class TestLoggerFactory:
    """Test the logger factory function"""
    
    def test_console_logger_creation(self):
        """Test console logger creation via factory"""
        logger = create_logger(
            backend="console",
            project_name="test_factory",
            experiment_name="test_console"
        )
        
        assert isinstance(logger, ConsoleLogger)
        assert logger.project_name == "test_factory"
        assert logger.experiment_name == "test_console"
        
        logger.finish()
    
    @patch('cs336_basics.logger.wandb')
    def test_wandb_logger_creation(self, mock_wandb):
        """Test W&B logger creation via factory"""
        mock_wandb.init.return_value = Mock()
        
        logger = create_logger(
            backend="wandb",
            project_name="test_factory",
            experiment_name="test_wandb",
            entity="test_entity"
        )
        
        assert isinstance(logger, WandbLogger)
        assert logger.project_name == "test_factory"
        
        logger.finish()
    
    @patch('cs336_basics.logger.wandb')
    def test_multi_logger_creation(self, mock_wandb):
        """Test multi-logger creation via factory"""
        mock_wandb.init.return_value = Mock()
        
        logger = create_logger(
            backend="multi",
            project_name="test_factory",
            experiment_name="test_multi"
        )
        
        assert isinstance(logger, MultiLogger)
        assert len(logger.loggers) == 2  # Should create both wandb and console
        
        logger.finish()
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error"""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_logger(backend="invalid", project_name="test")


class TestLoggerIntegration:
    """Integration tests for logger functionality"""
    
    def test_end_to_end_logging(self):
        """Test complete logging workflow"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            logger = ConsoleLogger(
                project_name="integration_test",
                experiment_name="end_to_end",
                log_dir=temp_dir
            )
            
            # Simulate a complete training workflow
            hyperparams = {
                "model": "transformer",
                "layers": 4,
                "lr": 3e-4,
                "batch_size": 32
            }
            logger.log_hyperparameters(hyperparams)
            
            # Create a small model for testing
            model = TransformerLM(
                vocab_size=100,
                context_length=32,
                d_model=64,
                num_layers=2,
                num_heads=4,
                d_ff=128
            )
            logger.log_model(model)
            
            # Simulate training steps
            for step in range(5):
                metrics = {
                    "train/loss": 3.0 - step * 0.1,
                    "train/accuracy": 0.1 + step * 0.05,
                    "learning_rate": 3e-4 * (0.95 ** step)
                }
                logger.log_metrics(metrics, step)
                
                if step % 2 == 0:
                    weights = torch.randn(100) * 0.1
                    logger.log_histogram("weights/layer1", weights, step)
            
            # Test optimizer logging
            optimizer = AdamW(model.parameters())
            logger.log_learning_rate(optimizer)
            
            # Add fake gradients and test gradient logging
            for param in model.parameters():
                param.grad = torch.randn_like(param) * 0.01
            
            grad_norms = logger.log_gradient_norms(model)
            assert "grad_norm/total" in grad_norms
            
            logger.finish()
            
            # Verify all expected files were created
            exp_dir = Path(temp_dir) / "integration_test" / "end_to_end"
            assert (exp_dir / "config.json").exists()
            assert (exp_dir / "metrics.json").exists()
            assert (exp_dir / "metrics.csv").exists()
            assert (exp_dir / "transformer_lm_architecture.txt").exists()
            assert (exp_dir / "metadata_final.json").exists()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_tests():
    """Run all tests"""
    print("Running experiment logging tests...")
    
    # Import pytest and run tests
    try:
        import pytest
        
        # Run tests with verbose output
        result = pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])
        
        if result == 0:
            print("✅ All experiment logging tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return result == 0
        
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Run basic tests manually
        try:
            # Test console logger
            temp_dir = tempfile.mkdtemp()
            logger = ConsoleLogger("test", "test", temp_dir)
            logger.log_hyperparameters({"test": True})
            logger.log_metrics({"loss": 1.0}, 1)
            logger.finish()
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print("✅ Basic console logger test passed!")
            
            # Test factory
            logger = create_logger("console", "test", "test")
            logger.finish()
            
            print("✅ Factory test passed!")
            print("✅ All basic tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Basic tests failed: {e}")
            return False


if __name__ == "__main__":
    run_tests()