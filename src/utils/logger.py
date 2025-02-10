import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
from typing import Dict, Any, Optional
import os
import yaml
from collections import OrderedDict
import datetime

class Logger:
    def __init__(self, config: Dict[str, Any], model: Optional[torch.nn.Module] = None):
        """
        Initialize logger with both TensorBoard and W&B support
        
        Args:
            config: Dictionary containing logging configuration
            model: Optional model to watch with W&B
        """
        logging_params = config.hyperparams_config.get('logging_params', {})
        self.use_wandb = logging_params.get('use_wandb', False)
        self.use_tensorboard = logging_params.get('use_tensorboard', False)
        
        # If no logging is enabled, return early
        if not (self.use_wandb or self.use_tensorboard):
            return
            
        self.current_metrics = {}
        self.model_watched = False
        
        # Create run name from timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract setting name from config
        setting_name = config.setting_config.get('setting_name', 'default')
        
        # Set up run name first
        exp_name = logging_params.get('exp_name', 'default')
        env_name = logging_params.get('env_name', 'default')
        self.run_name = f"{env_name}__{exp_name}__{timestamp}"
        
        # Set up logging directory only if needed
        if self.use_tensorboard:
            self.log_dir = self._setup_log_dir(config)
            
            # Initialize tensorboard
            self.writer = SummaryWriter(self.log_dir)
        
        # Initialize W&B if enabled
        if self.use_wandb and wandb.run is None:
            try:
                wandb.init(
                    project="inventory_control",
                    config=config.get_complete_config(),
                    name=f"inventory__{setting_name}__{timestamp}",
                    settings=wandb.Settings(start_method="thread")
                )
                print("Successfully initialized or connected to wandb")
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
        
        # Log hyperparameters only if logging is enabled
        if self.use_wandb or self.use_tensorboard:
            self._log_hyperparameters(config)
        
        print(f"Logging to: {self.log_dir}")  # Print the log directory for verification
        
        self.step_counter = 0  # Add a step counter
        if self.use_wandb:
            print(f"Config: {config}")
            if model is not None:
                self.model = model  # Store model reference
        self.model = model  # Store model reference
        self.best_metrics = {
            'train/loss/best': float('inf'),
            'dev/loss/best': float('inf'),
            'test/loss/best': float('inf')
        }

    def _setup_log_dir(self, config: Dict[str, Any]):
        # Create absolute path for logs
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = os.path.join(root_dir, "logs", "runs", self.run_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        return self.log_dir

    def _log_hyperparameters(self, config: Dict[str, Any]):
        """
        Log hyperparameters in an organized way to TensorBoard
        """
        # Define parameter groups
        param_groups = OrderedDict({
            "Problem Settings": [
                "n_stores", "n_warehouses", "n_extra_echelons", 
                "lost_demand", "maximize_profit", "is_hybrid"
            ],
            "Store Parameters": [
                "demand.distribution", "demand.mean", "lead_time.value",
                "holding_cost.value", "underage_cost.value", "procurement_cost.value"
            ],
            "Training Parameters": [
                "epochs", "batch_size", "learning_rate", "weight_decay",
                "gradient_clip"
            ],
            "Network Architecture": [
                "policy_network.hidden_layers", "policy_network.activation",
                "policy_network.dropout", "value_network.hidden_layers"
            ],
            "Dataset Parameters": [
                "train.n_samples", "train.periods", "dev.n_samples",
                "test.n_samples"
            ]
        })

        # Function to safely get nested dictionary values
        def get_nested_value(d, path):
            keys = path.split('.')
            value = d
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value

        # Build markdown text for TensorBoard
        markdown_text = "# Hyperparameters\n\n"
        
        for group_name, params in param_groups.items():
            markdown_text += f"## {group_name}\n"
            markdown_text += "|Parameter|Value|\n|-|-|\n"
            
            for param in params:
                value = get_nested_value(config, param)
                if value is not None:
                    markdown_text += f"|{param}|{value}|\n"
            
            markdown_text += "\n"

        # Add any custom discrete features if present
        if 'discrete_features' in config.setting_config.get('problem_params', {}):
            markdown_text += "## Discrete Features\n"
            markdown_text += "|Feature|Thresholds|Values|\n|-|-|-|\n"
            
            for feature, data in config.setting_config.get('problem_params', {}).get('discrete_features', {}).items():
                if isinstance(data, dict):  # Skip if None
                    thresholds = data.get('thresholds', [])
                    values = data.get('values', [])
                    markdown_text += f"|{feature}|{thresholds}|{values}|\n"

        # Log to TensorBoard
        self.writer.add_text("hyperparameters", markdown_text)

        # Save full config as YAML for reference
        config_path = os.path.join(self.log_dir, 'full_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def watch_model(self):
        """
        Start watching model after it's been initialized
        """
        if self.use_wandb and self.model is not None and not self.model_watched:
            try:
                wandb.watch(
                    self.model,
                    log="gradients",
                    log_freq=100,
                    log_graph=True
                )
                self.model_watched = True
            except Exception as e:
                print(f"Failed to watch model in W&B: {e}")

    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None, prefix: str = ""):
        """Store metrics and track best values"""
        if not (self.use_wandb or self.use_tensorboard):
            return
            
        # Update best metrics
        if f"{prefix}/loss/total" in metrics:
            current_loss = metrics[f"{prefix}/loss/total"]
            best_key = f"{prefix}/loss/best"
            self.best_metrics[best_key] = min(self.best_metrics[best_key], current_loss)
            self.current_metrics[best_key] = self.best_metrics[best_key]
        
        # Regular metric logging
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.current_metrics[name] = value
        
        if epoch is not None:
            self.current_metrics['epoch'] = epoch

    def log_model_weights(self, model: torch.nn.Module, step: int):
        """Log model weight histograms and statistics"""
        if not (self.use_wandb or self.use_tensorboard):
            return
            
        for name, param in model.named_parameters():
            # Log parameter values to tensorboard
            if param.data is not None:
                self.writer.add_histogram(f"weights/{name}", param.data, step)
                
                # Store statistics in current_metrics, using unbiased=False for std
                self.current_metrics[f"weights_stats/{name}_mean"] = param.data.mean().item()
                self.current_metrics[f"weights_stats/{name}_std"] = param.data.std(unbiased=False).item()
                self.current_metrics[f"weights_stats/{name}_norm"] = param.data.norm().item()
            
            # Log gradients if they exist
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, step)
                
                # Store gradient statistics in current_metrics, using unbiased=False for std
                self.current_metrics[f"grads_stats/{name}_mean"] = param.grad.mean().item()
                self.current_metrics[f"grads_stats/{name}_std"] = param.grad.std(unbiased=False).item()
                self.current_metrics[f"grads_stats/{name}_norm"] = param.grad.norm().item()
    
    def log_action_distribution(self, actions: torch.Tensor, step: int):
        """Log histogram of discrete actions"""
        if actions is not None:
            self.writer.add_histogram("actions/distribution", actions, step)
            # Calculate and store action frequencies
            unique_actions = torch.unique(actions)
            for action in unique_actions:
                freq = (actions == action).float().mean()
                self.current_metrics[f"actions/frequency_{action.item()}"] = freq.item()

    def flush_metrics(self):
        """
        Log all stored metrics to wandb and clear the current_metrics dictionary
        """
        if not self.use_wandb:
            return
            
        try:
            wandb.log(self.current_metrics)
        except Exception as e:
            print(f"Warning: Failed to log metrics to wandb: {e}")
        self.current_metrics = {}  # Clear the metrics after logging

    def close(self):
        """Close all logging connections"""
        if self.use_wandb:
            self.flush_metrics()
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Error closing wandb: {e}")
        
        if self.use_tensorboard:
            self.writer.close() 