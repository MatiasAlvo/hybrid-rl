import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
from typing import Dict, Any, Optional
import os
import yaml
from collections import OrderedDict
import datetime
import shutil  # Add this import for directory removal

class Logger:
    def __init__(self, config: Dict[str, Any], model: Optional[torch.nn.Module] = None):
        """
        Initialize logger with W&B support only
        
        Args:
            config: Dictionary containing logging configuration
            model: Optional model to watch with W&B
        """
        logging_params = config.hyperparams_config.get('logging_params', {})
        self.use_wandb = logging_params.get('use_wandb', False)
        # Remove tensorboard initialization entirely
        self.use_tensorboard = False
        
        # If no logging is enabled, return early
        if not self.use_wandb:
            return
            
        self.current_metrics = {}
        self.model_watched = False
        
        # Create run name from timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract setting name and problem parameters
        setting_name = logging_params.get('setting_name', 'default')
        problem_params = config.setting_config.get('problem_params', {})
        
        # Set up run name
        exp_name = logging_params.get('exp_name', 'default')
        env_name = logging_params.get('env_name', 'default')
        self.run_name = f"{env_name}__{exp_name}__{timestamp}"
        
        # Initialize W&B if enabled
        if self.use_wandb and wandb.run is None:
            try:
                wandb.init(
                    project="inventory_control",
                    config=config.get_complete_config(),
                    name=f"inventory__{setting_name}__{timestamp}",
                    group=setting_name,  # Use setting_name for grouping
                    settings=wandb.Settings(
                        start_method="thread",
                        # Disable local storage by setting dir to None
                        # This prevents wandb from creating local files
                        _disable_stats=True,
                        _disable_meta=True,
                        # sync_tensorboard=False,
                        save_code=False
                    )
                )
                
                # Add relevant problem parameters as tags
                wandb.run.tags = [
                    f"stores_{problem_params.get('n_stores', 0)}",
                    f"warehouses_{problem_params.get('n_warehouses', 0)}",
                    f"echelons_{problem_params.get('n_extra_echelons', 0)}",
                    'hybrid' if problem_params.get('is_hybrid', False) else 'single',
                    'profit' if problem_params.get('maximize_profit', False) else 'cost',
                    'lost_demand' if problem_params.get('lost_demand', False) else 'backorder'
                ]
                
                print("Successfully initialized or connected to wandb")
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
        
        # Log hyperparameters only if wandb is enabled
        if self.use_wandb:
            self._log_hyperparameters(config)
        
        self.step_counter = 0
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

    def _log_hyperparameters(self, config: Dict[str, Any]):
        """
        Log hyperparameters in an organized way to W&B
        """
        # Skip if wandb is not enabled
        if not self.use_wandb:
            return
            
        # Build markdown text for wandb
        markdown_text = "# Hyperparameters\n\n"
        
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

        for group_name, params in param_groups.items():
            markdown_text += f"## {group_name}\n"
            markdown_text += "|Parameter|Value|\n|-|-|\n"
            
            for param in params:
                value = get_nested_value(config, param)
                if value is not None:
                    markdown_text += f"|{param}|{value}|\n"
            
            markdown_text += "\n"

        # Add discrete features section if present
        if 'discrete_features' in config.setting_config.get('problem_params', {}):
            markdown_text += "## Discrete Features\n"
            markdown_text += "|Feature|Thresholds|Values|\n|-|-|-|\n"
            
            for feature, data in config.setting_config.get('problem_params', {}).get('discrete_features', {}).items():
                if isinstance(data, dict):
                    thresholds = data.get('thresholds', [])
                    values = data.get('values', [])
                    markdown_text += f"|{feature}|{thresholds}|{values}|\n"

        # Log to wandb only if enabled
        if self.use_wandb:
            wandb.log({"hyperparameters": markdown_text})
            # Log the full config directly to wandb instead of saving locally
            wandb.config.update(config.get_complete_config(), allow_val_change=True)

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
        if not self.use_wandb:
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
            # Store the metric, preserving wandb.Histogram objects
            self.current_metrics[name] = value
        
        if epoch is not None:
            self.current_metrics['epoch'] = epoch

    def log_model_weights(self, model: torch.nn.Module, step: int):
        """Log model weight histograms and statistics"""
        if not self.use_wandb:
            return
            
        for name, param in model.named_parameters():
            if param.data is not None:
                # Calculate statistics
                stats = {
                    f"weights_stats/{name}_mean": param.data.mean().item(),
                    f"weights_stats/{name}_std": param.data.std(unbiased=False).item(),
                    f"weights_stats/{name}_norm": param.data.norm().item()
                }
                self.current_metrics.update(stats)
                
                # Log histograms only if tensorboard is enabled
                if self.use_tensorboard:
                    self.writer.add_histogram(f"weights/{name}", param.data, step)
            
            # Handle gradients if they exist
            if param.grad is not None:
                # Calculate gradient statistics
                grad_stats = {
                    f"grads_stats/{name}_mean": param.grad.mean().item(),
                    f"grads_stats/{name}_std": param.grad.std(unbiased=False).item(),
                    f"grads_stats/{name}_norm": param.grad.norm().item()
                }
                self.current_metrics.update(grad_stats)
                
                # Log gradient histograms only if tensorboard is enabled
                if self.use_tensorboard:
                    self.writer.add_histogram(f"grads/{name}", param.grad, step)

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
            # Log metrics to wandb
            wandb.log(self.current_metrics)
        except Exception as e:
            print(f"Warning: Failed to log metrics to wandb: {e}")
        
        # Clear the metrics after logging
        self.current_metrics = {}

    def close(self):
        """Close wandb connection and delete local files"""
        if self.use_wandb:
            self.flush_metrics()
            try:
                # Get the wandb run directory before finishing
                if wandb.run is not None:
                    wandb_dir = wandb.run.dir
                    
                    # Finish the run
                    wandb.finish()
                    
                    # Delete the wandb directory if it exists
                    if os.path.exists(wandb_dir):
                        print(f"Cleaning up wandb directory: {wandb_dir}")
                        try:
                            shutil.rmtree(wandb_dir)
                        except Exception as e:
                            print(f"Warning: Failed to delete wandb directory: {e}")
                    
                    # Also try to clean up the wandb folder in the current directory
                    wandb_folder = os.path.join(os.getcwd(), "wandb")
                    if os.path.exists(wandb_folder):
                        print(f"Cleaning up wandb folder: {wandb_folder}")
                        try:
                            shutil.rmtree(wandb_folder)
                        except Exception as e:
                            print(f"Warning: Failed to delete wandb folder: {e}")
                else:
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Error closing wandb: {e}")

    def log_image(self, name: str, image, step: Optional[int] = None):
        """
        Log an image to wandb
        
        Args:
            name: Name of the image
            image: PIL Image or numpy array
            step: Optional step number
        """
        if not self.use_wandb:
            return
        
        try:
            # Log image to wandb
            self.current_metrics[name] = wandb.Image(image)
            
            # If step is provided, log immediately
            if step is not None:
                wandb.log({name: wandb.Image(image), 'epoch': step})
        except Exception as e:
            print(f"Warning: Failed to log image to wandb: {e}") 