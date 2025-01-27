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
        # Create a more specific run name including timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config.logging_params.get('env_name', 'inventory')}__{config.logging_params.get('exp_name', 'experiment')}__{timestamp}"
        
        # Create absolute path for logs
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = os.path.join(root_dir, "logs", "runs", self.run_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard with absolute path
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        print(f"Logging to: {self.log_dir}")  # Print the log directory for verification
        
        # Log hyperparameters in an organized way
        self._log_hyperparameters(config)
        
        # Initialize W&B if enabled
        self.use_wandb = config.logging_params.get('use_wandb', False)
        print(f"W&B enabled: {self.use_wandb}")  # Debug print
        
        self.step_counter = 0  # Add a step counter
        self.current_metrics = {}  # Add a dictionary to store metrics
        if self.use_wandb:
            print(f"Initializing W&B with project: {config.logging_params.get('wandb_project_name', 'inventory_control')}, entity: {config.logging_params.get('wandb_entity', None)}")
            print(f"Config: {config}")
            wandb.init(
                project=config.logging_params.get('wandb_project_name', 'inventory_control'),
                sync_tensorboard=True,
                config=config,
                name=self.run_name,
                monitor_gym=False,  # Disable gym monitoring
                save_code=True,
            )
            if model is not None:
                self.model = model  # Store model reference
        self.model = model  # Store model reference
        self.model_watched = False  # Flag to track if model has been watched

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
        if 'discrete_features' in config.problem_params:
            markdown_text += "## Discrete Features\n"
            markdown_text += "|Feature|Thresholds|Values|\n|-|-|-|\n"
            
            for feature, data in config.problem_params['discrete_features'].items():
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
        """
        Store metrics in the current_metrics dictionary
        """
        # Try to watch model if not already watching
        if not self.model_watched:
            self.watch_model()
            
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.writer.add_scalar(name, value, epoch)  # Still log to tensorboard immediately
            self.current_metrics[name] = value
            self.current_metrics['epoch'] = epoch
    
    def log_model_weights(self, model: torch.nn.Module, step: int):
        """Log model weight histograms and statistics"""
        for name, param in model.named_parameters():
            # Log parameter values to tensorboard
            if param.data is not None:
                self.writer.add_histogram(f"weights/{name}", param.data, step)
                
                # Store statistics in current_metrics
                self.current_metrics[f"weights_stats/{name}_mean"] = param.data.mean().item()
                self.current_metrics[f"weights_stats/{name}_std"] = param.data.std().item()
                self.current_metrics[f"weights_stats/{name}_norm"] = param.data.norm().item()
            
            # Log gradients if they exist
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, step)
                
                # Store gradient statistics in current_metrics
                self.current_metrics[f"grads_stats/{name}_mean"] = param.grad.mean().item()
                self.current_metrics[f"grads_stats/{name}_std"] = param.grad.std().item()
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
        if self.use_wandb and self.current_metrics:
            wandb.log(self.current_metrics)
            self.current_metrics = {}  # Clear the metrics after logging

    def close(self):
        """Close all logging connections"""
        self.flush_metrics()  # Ensure any remaining metrics are logged
        self.writer.close()
        if self.use_wandb:
            wandb.finish() 