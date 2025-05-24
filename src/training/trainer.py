from src import torch, logging, Path, np
from src.envs.inventory.env import InventoryEnv
from src.algorithms.hdpo.collectors.collector import InventoryCollector
from src.algorithms.hdpo.losses.pathwise import HDPOLoss
from src.envs.base_env import BaseEnvironment
from src.algorithms.base import BaseAlgorithm
from src.data.data_handling import Dataset
from typing import Dict, Optional, Tuple
import os
import copy
import datetime
import matplotlib.pyplot as plt
from src.utils.logger import Logger
import yaml
import pandas as pd
import wandb


class Trainer():
    """
    Trainer class
    """

    def __init__(self,  device='cpu'):
        
        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = [] 
        self.device = device
        self.time_stamp = self.get_time_stamp()
        self.best_performance_data = {'train_loss': np.inf, 'dev_loss': np.inf, 'last_epoch_saved': -1000, 'model_params_to_save': None}
        self.logger = None  # Initialize logger as None
    
    def reset(self):
        """
        Reset the losses
        """

        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = []

    def train(self, epochs, loss_function, simulator, model, data_loaders, 
              optimizer_wrapper, problem_params, observation_params, 
              params_by_dataset, trainer_params, config):
        """Training loop using optimizer_wrapper for parameter updates"""
        
        # Initialize logger only if logging is enabled in config
        logging_params = config.hyperparams_config.get('logging_params', {})
        if logging_params.get('use_wandb', False) or logging_params.get('use_tensorboard', False):
            self.logger = Logger(config, model)
        else:
            self.logger = None
            
        global_step = 0
        
        # Get learning rate annealing parameters
        optimizer_params = config.hyperparams_config.get('optimizer_params', {})
        initial_lr = optimizer_params.get('learning_rate', 0.0003)
        anneal_lr = optimizer_params.get('anneal_lr', False)
        
        for epoch in range(epochs):
            # Update learning rate if annealing is enabled
            if anneal_lr:
                frac = 1.0 - (epoch / epochs)
                new_lr = frac * initial_lr
                for param_group in optimizer_wrapper.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                # Log LR if logger exists
                if self.logger is not None:
                    self.logger.log_metrics({'train/learning_rate': new_lr}, epoch)
            
            # Training epoch
            train_metrics = self.do_one_epoch(
                optimizer_wrapper,
                data_loaders['train'],
                loss_function,
                simulator,
                model,
                params_by_dataset['train']['periods'],
                problem_params,
                observation_params,
                train=True,
                ignore_periods=params_by_dataset['train']['ignore_periods']
            )
            
            # After first forward pass, model should be initialized
            if self.logger is not None:
                self.logger.watch_model()  # This will only take effect once
            
            # Validation epoch
            with torch.no_grad():
                dev_metrics, trajectory_data, additional_data = self.do_one_epoch(
                    optimizer_wrapper,
                    data_loaders['dev'],
                    loss_function,
                    simulator,
                    model,
                    params_by_dataset['dev']['periods'],
                    problem_params,
                    observation_params,
                    train=False,
                    ignore_periods=params_by_dataset['dev']['ignore_periods'],
                    return_trajectory=True,
                    collect_additional_data=True
                )
            
            # Only log if logger exists
            if self.logger is not None:
                self.logger.log_metrics(train_metrics, epoch, prefix='train')
                self.logger.log_metrics(dev_metrics, epoch, prefix='dev')
                self.logger.log_model_weights(model, epoch)
                
                if 'actions' in train_metrics:
                    self.logger.log_action_distribution(train_metrics['actions'], epoch)
                
                # Generate and log inventory vs action plot for dev set with dev loss
                if 'trajectory_data' in dev_metrics:
                    self.log_inventory_action_plot(
                        dev_metrics['trajectory_data'], 
                        epoch, 
                        dev_loss=dev_metrics['loss/reported']
                    )
                
                self.logger.flush_metrics()
            
            # Update best parameters and save if needed
            self.update_best_params_and_save(
                epoch, 
                train_metrics['loss/total'], 
                dev_metrics['loss/total'],
                trainer_params, 
                model, 
                optimizer_wrapper.optimizer
            )
            
            # Log progress
            if epoch % trainer_params['print_results_every_n_epochs'] == 0:
                print(f'Epoch {epoch}: Train Loss = {train_metrics["loss/reported"]:.4f}, '
                      f'Dev Loss = {dev_metrics["loss/reported"]:.4f}')
        # log best train loss and dev loss, if there is a logger
        if self.logger is not None:
            self.logger.log_metrics({'train/loss/best': self.best_performance_data['train_loss'], 'dev/loss/best': self.best_performance_data['dev_loss']})
            self.logger.flush_metrics()
        return train_metrics, dev_metrics

    def test(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params, discrete_allocation=False):
        """Test the model using the best parameters found during training"""
        if model.trainable:
            if self.best_performance_data['model_params_to_save'] is not None:
                try:
                    model.load_state_dict(self.best_performance_data['model_params_to_save'])
                    print(f"Loaded best model with dev loss: {self.best_performance_data['dev_loss']:.4f}")
                except RuntimeError as e:
                    print(f"Error: Failed to load model state dict: {e}")
                    raise
            else:
                print("Warning: No best model parameters found. Using current model state.")

        # Put model in eval mode
        model.eval()

        test_metrics, trajectory_data, additional_data = self.do_one_epoch(
                optimizer, 
                data_loaders['test'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['test']['periods'], 
                problem_params, 
                observation_params, 
                train=False,
                ignore_periods=params_by_dataset['test']['ignore_periods'],
                discrete_allocation=discrete_allocation,
                return_trajectory=True,
                collect_additional_data=trainer_params.get('compute_metrics_on_test', False)
                )
        
        # Log only if logger exists
        if self.logger is not None:
            scalar_metrics = {
                'loss/reported': test_metrics['loss/reported'],
                'loss/total': test_metrics['loss/total']
            }
            self.logger.log_metrics(scalar_metrics, prefix='test')
            self.logger.flush_metrics()
        
        # Compute additional metrics if specified
        if trainer_params.get('compute_metrics_on_test', False):
            self.compute_and_save_test_metrics(
                trajectory_data,
                additional_data,
                model_name=trainer_params['save_model_filename'],
                folders=trainer_params['save_model_folders'],
                simulator=simulator
            )
        
        # Put model back in train mode
        model.train()
        
        return test_metrics, trajectory_data

    def do_one_epoch(self, optimizer_wrapper, data_loader, loss_function, simulator, model, periods, problem_params, observation_params, train=True, ignore_periods=0, discrete_allocation=False, return_trajectory=False, collect_additional_data=False):
        """
        Do one epoch of training or testing
        """
        
        epoch_loss = 0
        epoch_loss_to_report = 0
        total_samples = len(data_loader.dataset)
        periods_tracking_loss = periods - ignore_periods
        
        optimizer_metrics_sum = None
        num_batches = 0
        special_metrics = {}  # New dictionary for metrics that shouldn't be averaged
        
        # Initialize trajectory and additional data
        trajectory_data = None
        additional_data = None if not collect_additional_data else {}
        
        # Initialize data structures for action distribution histograms
        action_histograms = {
            'discrete_probs': [],
            'discrete_logits': [],
            'pre_temp_logits': []
        }

        for i, data_batch in enumerate(data_loader):
            data_batch = self.move_batch_to_device(data_batch)
            
            # Forward pass and simulation - pass train parameter
            total_reward, reward_to_report, batch_trajectory_data, batch_additional_data, batch_action_data = self.simulate_batch(
                loss_function, simulator, model, periods, problem_params, data_batch, observation_params, 
                ignore_periods, discrete_allocation, collect_trajectories=True, train=train,
                collect_additional_data=collect_additional_data
            )
            
            
            # Collect action distribution data for histograms
            if batch_action_data:
                for key in action_histograms:
                    if key in batch_action_data and batch_action_data[key] is not None:
                        # Extract only the last index (order action) for each sample
                        # Assuming the shape is [batch_size, num_actions]
                        order_action_data = batch_action_data[key][:, 0, -1]  # Get last index for each sample
                        action_histograms[key].append(order_action_data)
            
            # Always accumulate simulator metrics
            epoch_loss += total_reward.item()
            epoch_loss_to_report += reward_to_report.item()
            
            # If training, get optimizer metrics but don't use them for loss tracking
            if train and model.trainable:
                batch_metrics = optimizer_wrapper.optimize(batch_trajectory_data)
                
                # Handle special metrics (like histograms) that shouldn't be averaged
                for k, v in batch_metrics.items():
                    if 'histogram' in k or isinstance(v, wandb.Histogram):
                        special_metrics[k] = v  # Store without averaging
                        continue
                        
                    if optimizer_metrics_sum is None:
                        optimizer_metrics_sum = {k: v for k, v in batch_metrics.items() 
                                               if 'histogram' not in k and not isinstance(v, wandb.Histogram)}
                    else:
                        for k, v in batch_metrics.items():
                            if 'histogram' not in k and not isinstance(v, wandb.Histogram):
                                optimizer_metrics_sum[k] += v
                    num_batches += 1
            
            # Store trajectory data for the first batch only
            if return_trajectory and trajectory_data is None:
                trajectory_data = batch_trajectory_data
                
            # Store additional data for the first batch only
            if collect_additional_data and batch_additional_data is not None:
                if additional_data is None:
                    additional_data = {}
                for k, v in batch_additional_data.items():
                    if k not in additional_data:
                        additional_data[k] = v

        # Calculate average metrics using simulator results
        metrics = {
            'loss/total': epoch_loss/(total_samples*periods*problem_params['n_stores']),
            'loss/reported': epoch_loss_to_report/(total_samples*periods_tracking_loss*problem_params['n_stores'])
        }
        
        # Add averaged optimizer metrics
        if optimizer_metrics_sum is not None:
            for k, v in optimizer_metrics_sum.items():
                metrics[k] = v / num_batches
                
        # Add special metrics without averaging
        metrics.update(special_metrics)
        
        # Process action distribution histograms
        for key in action_histograms:
            if action_histograms[key]:
                # Concatenate all tensors for this key
                try:
                    all_data = torch.cat(action_histograms[key], dim=0)
                    # Create wandb histogram
                    metrics[f'action_distribution/{key}'] = wandb.Histogram(all_data.flatten().cpu().numpy())
                except Exception as e:
                    print(f"Error creating histogram for {key}: {e}")
        
        # Include trajectory data in metrics for visualization
        if not train and trajectory_data is not None:
            metrics['trajectory_data'] = trajectory_data

        if return_trajectory:
            if collect_additional_data:
                return metrics, trajectory_data, additional_data
            else:
                return metrics, trajectory_data, None
        else:
            return metrics
    
    def simulate_batch(self, loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods=0, discrete_allocation=False, collect_trajectories=False, train=True, collect_additional_data=False):
        """
        Simulate for an entire batch of data, across the specified number of periods.
        Collects data for both HDPO (pathwise gradients) and optionally PPO (trajectory data).
        """
        # Initialize rewards
        batch_reward = 0
        reward_to_report = 0

        # Get observation keys from value network config if it exists
        observation_keys = self._get_observation_keys(model)

        # Initialize data collection structures
        trajectory_data = self._initialize_trajectory_data(collect_trajectories)
        additional_data = self._initialize_additional_data(collect_additional_data)
        
        # Initialize action distribution data collection
        action_data = {
            'discrete_probs': None,
            'discrete_logits': None,
            'pre_temp_logits': None
        }

        # Reset simulator
        observation, _ = simulator.reset(periods, problem_params, data_batch, observation_params)
        
        for t in range(periods):
            # Store observation if collecting trajectories
            vectorized_obs = self.vectorize_observation(observation, observation_keys)

            # Add internal data to observation
            observation_and_internal_data = self._prepare_observation_with_internal_data(observation, simulator)

            # Sample action and get policy outputs
            model_output = model(observation_and_internal_data, train=train)
            action_dict = model_output.get('action_dict')
            raw_outputs = model_output.get('raw_outputs', {})
            value = model_output.get('value', None)
            
            # Collect action distribution data (only from the last period)
            if t == periods - 1:
                if 'discrete_probs' in action_dict:
                    action_data['discrete_probs'] = action_dict['discrete_probs'].detach().clone()
                if 'discrete' in raw_outputs:
                    action_data['discrete_logits'] = raw_outputs['discrete'].detach().clone()
                if 'pre_temp_discrete_logits' in raw_outputs:
                    action_data['pre_temp_logits'] = raw_outputs['pre_temp_discrete_logits'].detach().clone()
            
            # Collect additional data if requested
            if collect_additional_data:
                self._collect_additional_data(additional_data, model_output, action_dict)

            # Apply discrete allocation if needed
            if discrete_allocation:
                action_dict = self._apply_discrete_allocation(action_dict)

            # Execute environment step
            next_observation, reward, terminated, _, _ = simulator.step(observation, action_dict)
            total_reward = loss_function(None, action_dict, reward)

            # Collect trajectory data if requested
            if collect_trajectories:
                self._collect_trajectory_data(trajectory_data, vectorized_obs, action_dict, value, reward, terminated)

            # Update running rewards
            batch_reward += total_reward
            if t >= ignore_periods:
                reward_to_report += total_reward

            # Update observation
            observation = next_observation

            if terminated:
                break

        # Process collected data
        trajectory_data = self._process_trajectory_data(trajectory_data, collect_trajectories)
        additional_data = self._process_additional_data(additional_data, collect_additional_data)
        
        # Add final observation to trajectory data
        if collect_trajectories and trajectory_data is not None:
            trajectory_data['next_observation'] = observation

        return batch_reward, reward_to_report, trajectory_data, additional_data, action_data

    def _get_observation_keys(self, model):
        """Extract observation keys from model if available"""
        if hasattr(model, 'value_net') and model.value_net is not None:
            return model.value_net.observation_keys
        return None

    def _initialize_trajectory_data(self, collect_trajectories):
        """Initialize trajectory data structure if needed"""
        if not collect_trajectories:
            return None
        
        return {
            'observations': [],      # observations at each time step
            'rewards': [],           # rewards at each time step
            'discrete_action_indices': [],  # one action per sub-range
            'total_action': [],      # total one-dimensional action
            'log_probs': [],            # log_probs for the selected sub-range
            'values': [],            # value of the state
            'terminated': []         # termination flags
        }

    def _initialize_additional_data(self, collect_additional_data):
        """Initialize additional data structure if needed"""
        if not collect_additional_data:
            return None
        
        return {}

    def _prepare_observation_with_internal_data(self, observation, simulator):
        """Add internal data to observation"""
        observation_and_internal_data = {k: v for k, v in observation.items()}
        observation_and_internal_data['internal_data'] = simulator._internal_data
        return observation_and_internal_data

    def _collect_additional_data(self, additional_data, model_output, action_dict):
        """Collect additional data from model outputs and action dictionary"""
        # Store raw outputs from model
        if 'raw_outputs' in model_output:
            raw_outputs = model_output['raw_outputs']
            for key, value in raw_outputs.items():
                if value is not None:
                    if key not in additional_data:
                        additional_data[key] = []
                    additional_data[key].append(value.detach().clone())
        
        # Store continuous values from action_dict if they exist
        if 'continuous_values' in action_dict:
            if 'continuous_values' not in additional_data:
                additional_data['continuous_values'] = []
            additional_data['continuous_values'].append(action_dict['continuous_values'].detach().clone())

    def _apply_discrete_allocation(self, action_dict):
        """Apply discrete allocation by rounding action values"""
        return {key: val.round() for key, val in action_dict.items()}

    def _collect_trajectory_data(self, trajectory_data, vectorized_obs, action_dict, value, reward, terminated):
        """Collect trajectory data for the current step"""
        if vectorized_obs is not None:
            trajectory_data['observations'].append(vectorized_obs.clone())
        
        # Only append fields that exist and are not None
        if 'discrete_action_indices' in action_dict and action_dict['discrete_action_indices'] is not None:
            trajectory_data['discrete_action_indices'].append(action_dict['discrete_action_indices'].detach().clone())
        
        if 'feature_actions' in action_dict and 'total_action' in action_dict['feature_actions']:
            trajectory_data['total_action'].append(action_dict['feature_actions']['total_action'].detach().clone())
        
        # Handle log_probs - check if they exist and are not None
        if 'log_probs' in action_dict and action_dict['log_probs'] is not None:
            if 'log_probs' not in trajectory_data:
                trajectory_data['log_probs'] = []
            trajectory_data['log_probs'].append(action_dict['log_probs'].detach().clone())
        else:
            # For agents that don't use log_probs (like ContinuousOnly), add a dummy tensor or None
            if 'log_probs' not in trajectory_data:
                trajectory_data['log_probs'] = []
            trajectory_data['log_probs'].append(None)
        
        if value is not None:
            trajectory_data['values'].append(value.detach().clone())
        
        trajectory_data['rewards'].append(reward.clone())
        trajectory_data['terminated'].append(torch.tensor(terminated).detach().clone())
        
        # Save raw_continuous_samples if they exist (for GaussianPPOAgent)
        if 'raw_continuous_samples' in action_dict:
            if 'raw_continuous_samples' not in trajectory_data:
                trajectory_data['raw_continuous_samples'] = []
            trajectory_data['raw_continuous_samples'].append(action_dict['raw_continuous_samples'].detach().clone())

    def _process_trajectory_data(self, trajectory_data, collect_trajectories):
        """Process collected trajectory data into tensors"""
        if not collect_trajectories or trajectory_data is None:
            return trajectory_data
        
        processed_data = {}
        for k, v in trajectory_data.items():
            if not v:
                processed_data[k] = None
            elif v[0] is None:
                # Handle lists containing None values
                processed_data[k] = None
            else:
                processed_data[k] = torch.stack(v)
        
        return processed_data

    def _process_additional_data(self, additional_data, collect_additional_data):
        """Process collected additional data into tensors"""
        if not collect_additional_data or additional_data is None:
            return additional_data
        
        return {
            k: torch.stack(v) if v and v[0] is not None else None 
            for k, v in additional_data.items()
        }

    def save_model(self, epoch, model, optimizer, trainer_params):
        path = self.create_many_folders_if_not_exist_and_return_path(
            base_dir=trainer_params['base_dir'], 
            intermediate_folder_strings=trainer_params['save_model_folders']
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_loss': self.best_performance_data['train_loss'],
            'best_dev_loss': self.best_performance_data['dev_loss'],
            'all_train_losses': self.all_train_losses,
            'all_dev_losses': self.all_dev_losses,
            'all_test_losses': self.all_test_losses,
        }
        
        torch.save(checkpoint, f"{path}/{trainer_params['save_model_filename']}.pt")

    def create_folder_if_not_exists(self, folder):
        """
        Create a directory in the corresponding file, if it does not already exist
        """

        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    def create_many_folders_if_not_exist_and_return_path(self, base_dir, intermediate_folder_strings):
        """
        Create a directory in the corresponding file for each file in intermediate_folder_strings, if it does not already exist
        """

        path = base_dir
        for string in intermediate_folder_strings:
            path += f"/{string}"
            self.create_folder_if_not_exists(path)
        return path
    
    def update_best_params_and_save(self, epoch, train_loss, dev_loss, trainer_params, model, optimizer):
        """
        Update best model parameters if it achieves best performance so far, and save the model
        """
        data_for_compare = {'train_loss': train_loss, 'dev_loss': dev_loss}
        if data_for_compare[trainer_params['choose_best_model_on']] < self.best_performance_data[trainer_params['choose_best_model_on']]:  
            self.best_performance_data['train_loss'] = train_loss
            self.best_performance_data['dev_loss'] = dev_loss
            if model.trainable:
                # Save the entire model's state dict instead of just the policy
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.state_dict())
            self.best_performance_data['update'] = True

        if trainer_params['save_model'] and model.policy.trainable:
            if self.best_performance_data['last_epoch_saved'] + trainer_params['epochs_between_save'] <= epoch and self.best_performance_data['update']:
                self.best_performance_data['last_epoch_saved'] = epoch
                self.best_performance_data['update'] = False
                self.save_model(epoch, model, optimizer, trainer_params)
    
    def plot_losses(self, ymin=None, ymax=None):
        """
        Plot train and test losses for each epoch
        """

        plt.plot(self.all_train_losses, label='Train loss')
        plt.plot(self.all_dev_losses, label='Dev loss')
        plt.legend()

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def move_batch_to_device(self, data_batch):
        """
        Move a batch of data to the device (CPU or GPU)
        """

        return {k: v.to(self.device) for k, v in data_batch.items()}
    
    def load_model(self, model, optimizer_wrapper, model_path):
        """Load a saved model"""
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_wrapper.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.all_train_losses = checkpoint['all_train_losses']
        self.all_dev_losses = checkpoint['all_dev_losses']
        self.all_test_losses = checkpoint['all_test_losses']
        return model, optimizer_wrapper
    
    def get_time_stamp(self):

        return int(datetime.datetime.now().timestamp())
    
    def get_year_month_day(self):
        """"
        Get current date in year_month_day format
        """

        ct = datetime.datetime.now()
        return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"

    def vectorize_observation(self, observation, observation_keys=None):
        """
        Convert an observation dictionary into a flat vector based on specified keys.
        
        Parameters:
        -----------
        observation: dict
            The observation dictionary to vectorize
        observation_keys: list, optional
            List of keys from observation to include in vectorization
        """
        if observation_keys is None:
            # Default behavior - only track store inventories
            return observation['store_inventories'].reshape(observation['store_inventories'].shape[0], -1).clone()  # Use clone to ensure the return is frozen
        
        vectors = []
        
        # Build vector using specified keys
        for key in observation_keys:
            if key in observation:
                to_append = observation[key]
                if to_append.shape[0] != observation['store_inventories'].shape[0]:
                    to_append = to_append.expand(observation['store_inventories'].shape[0], -1)
                vectors.append(to_append.reshape(to_append.shape[0], -1).clone().to(self.device))  # Use clone to ensure the return is frozen
        
        if not vectors:  # If nothing to track, return None
            return None
        
        return torch.cat(vectors, dim=-1).clone()  # Detach and clone to ensure the return is frozen

    def vectorize_action(self, action_dict):
        """
        Convert an action dictionary into a flat vector.
        Order is important and must be consistent for both vectorization and de-vectorization.
        """
        return action_dict['log_probs'].reshape(action_dict['log_probs'].shape[0], -1).detach()

    def compute_loss(self, trajectory_dict, loss_function):
        """
        Compute loss based on trajectory and cost structure
        """
        costs = trajectory_dict['costs']
        
        if isinstance(costs, dict) and 'total' in costs:
            # Use total cost for loss computation
            return loss_function(trajectory_dict, costs['total'])
        else:
            # Legacy behavior for simple cost structure
            return loss_function(trajectory_dict, costs)
    def _collect_trajectories(self, model, simulator):
        """Collect trajectories using model and simulator"""
        # Get agent outputs
        agent_outputs = model(simulator.get_observation())
        
        # Simulate using simulator_actions
        simulator_info = simulator.step(agent_outputs['simulator_actions'])
        
        # Combine all info needed for optimization
        return {
            'raw_outputs': agent_outputs['raw_outputs'],
            'probabilities': agent_outputs['probabilities'],
            'value': agent_outputs['value'],
            'actions_per_range': agent_outputs['actions_per_range'],
            'actions_per_feature': agent_outputs['actions_per_feature'],
            'simulator_info': simulator_info
        }

    def compute_and_save_test_metrics(self, trajectory_data, additional_data, model_name, folders, simulator, n_samples=100):
        """
        Compute and save specific metrics for a random subset of test trajectories
        """
        # Create metrics directory
        base_dir = 'metrics/test_trajectories'
        path = self.create_many_folders_if_not_exist_and_return_path(
            base_dir=base_dir,
            intermediate_folder_strings=folders
        )
        
        # Get shapes: [T, B, F] for observations, [T, B, 1] for actions
        T, B, _ = trajectory_data["observations"].shape
        
        # Select random batch indices first
        random_batch_indices = torch.randperm(B)[:n_samples]
        
        # Select the samples for each tensor
        selected_inventories = trajectory_data["observations"][:, random_batch_indices, :]  # Shape: [T, n_samples, F]
        selected_discrete_action_indices = trajectory_data["discrete_action_indices"][:, random_batch_indices, :]  # Shape: [T, n_samples, 1]
        selected_total_action = trajectory_data["total_action"][:, random_batch_indices, :]  # Shape: [T, n_samples, 1]
        
        # Select additional data if available
        selected_additional_data = {}
        if additional_data:
            for key, tensor in additional_data.items():
                if tensor is not None:
                    selected_additional_data[key] = tensor[:, random_batch_indices]
        
        # if we are normalizing the inventory, we need to unnormalize selected_inventories
        if simulator.normalize_observations:
            selected_inventories = selected_inventories * simulator.inventory_std + simulator.inventory_mean
        
        # Create DataFrame
        all_data = []
        
        # Loop through time steps and batch samples
        for t in range(T):
            for b_idx, b in enumerate(random_batch_indices[:n_samples]):
                # Get inventory and action data
                inventory = selected_inventories[t, b_idx].detach().cpu().numpy()
                discrete_action_idx = selected_discrete_action_indices[t, b_idx, 0].item()
                total_action = selected_total_action[t, b_idx, 0].item()
                
                # Create base record
                record = {
                    'time_step': t,
                    'batch_idx': b.item(),
                    'inventory_on_hand': inventory[0],
                    'inventory_sum': inventory.sum(),
                    'discrete_action_index': discrete_action_idx,
                    'total_action': total_action
                }
                
                # Add additional data as tuples
                for key, tensor in selected_additional_data.items():
                    # For a single store problem, we can just take the first store
                    # Shape is typically [T, n_samples, stores, features]
                    # We want to extract [t, b_idx, 0, :] and convert to tuple
                    try:
                        # Try to access the first store dimension (index 0)
                        # This works for tensors with shape [T, B, stores, features]
                        data_array = tensor[t, b_idx, 0].detach().cpu().numpy()
                    except IndexError:
                        # If that fails, the tensor might not have a store dimension
                        # Try without the store dimension
                        try:
                            data_array = tensor[t, b_idx].detach().cpu().numpy()
                        except IndexError:
                            # If that also fails, skip this tensor
                            continue
                    
                    # Convert to tuple and add to record
                    record[key] = tuple(data_array.flatten())
                
                all_data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save as CSV
        csv_path = f"{path}/{model_name}_test_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved test data to {csv_path}")

    def log_inventory_action_plot(self, trajectory_data, epoch, dev_loss=None):
        """
        Generate and log a plot showing the relationship between inventory and actions to wandb.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            epoch: Current epoch number
            dev_loss: Optional dev loss to include in the title
        """
        try:
            # Skip if logger is not available
            if self.logger is None or not hasattr(self.logger, 'use_wandb') or not self.logger.use_wandb:
                return
            
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image
            import wandb
            
            # Check if we have the necessary data
            if "observations" not in trajectory_data or "total_action" not in trajectory_data:
                print("Missing required data for inventory-action plot")
                return
            
            # Get shapes: [T, B, F] for observations, [T, B, 1] for actions
            T, B, _ = trajectory_data["observations"].shape
            
            # Limit the number of samples to plot (reduced from 100 to 50)
            n_samples = min(30, B)
            
            # Select random batch indices first
            random_batch_indices = torch.randperm(B)[:n_samples]
            
            # Select the samples for each tensor
            selected_inventories = trajectory_data["observations"][:, random_batch_indices, :]  # Shape: [T, n_samples, F]
            selected_total_action = trajectory_data["total_action"][:, random_batch_indices, :]  # Shape: [T, n_samples, 1]
            
            # Get discrete action indices if available
            has_discrete_actions = "discrete_action_indices" in trajectory_data
            if has_discrete_actions:
                selected_discrete_actions = trajectory_data["discrete_action_indices"][:, random_batch_indices, :]  # Shape: [T, n_samples, 1]
                # Flatten and convert to numpy
                discrete_actions_flat = selected_discrete_actions.reshape(-1).detach().cpu().numpy()
            
            # Flatten time and batch dimensions for plotting
            # Reshape to [T*n_samples, F] and [T*n_samples, 1]
            inventories_flat = selected_inventories.reshape(-1, selected_inventories.shape[-1]).detach().cpu()
            actions_flat = selected_total_action.reshape(-1, 1).detach().cpu()
            
            # Calculate inventory sums (sum across feature dimension)
            inventory_sum = inventories_flat.sum(dim=1).numpy()
            total_action = actions_flat.squeeze().numpy()
            
            # Create the plot with larger figure size
            plt.figure(figsize=(16, 10))
            
            # Color points by discrete action if available
            if has_discrete_actions:
                # Get unique discrete actions for coloring
                unique_actions = np.unique(discrete_actions_flat)
                
                # Create a colormap with distinct colors - Fix deprecated get_cmap
                if len(unique_actions) <= 10:
                    # For few actions, use tab10 colormap - using the new recommended approach
                    cmap = plt.colormaps['tab10']
                else:
                    # For many actions, use hsv colormap - using the new recommended approach
                    cmap = plt.colormaps['hsv']
                
                # Plot each discrete action with a different color
                for i, action in enumerate(unique_actions):
                    # Fix boolean mask conversion warning by explicitly converting to integer indices
                    mask = np.where(discrete_actions_flat == action)[0]
                    plt.scatter(
                        inventory_sum[mask], 
                        total_action[mask], 
                        alpha=0.8, 
                        s=30,  # Larger points for better visibility
                        color=cmap(i % cmap.N),  # Use modulo to ensure we don't exceed colormap range
                        label=f'Action {int(action)}'
                    )
            else:
                # If no discrete actions, use a single color
                plt.scatter(inventory_sum, total_action, alpha=0.7, s=50)
            
            # Add (s, S) policy line with thicker line
            s, S = 26, 62  # Example values, adjust as needed
            inventory_range = np.linspace(np.min(inventory_sum), np.max(inventory_sum), 100)
            order_amounts = np.maximum(S - inventory_range, 0) * (inventory_range <= s)
            plt.plot(inventory_range, order_amounts, color='black', linewidth=3, label='(s, S) Policy')
            
            # Add dev loss to title if provided
            title = f'Inventory vs Actions (Epoch {epoch})'
            if dev_loss is not None:
                title += f' - Dev Loss: {dev_loss:.4f}'
            
            # Use larger font sizes for better readability
            plt.title(title, fontsize=18)
            plt.xlabel('Total Inventory', fontsize=16)
            plt.ylabel('Total Actions', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            # Add legend with reasonable size and position
            if has_discrete_actions and len(unique_actions) <= 10:  # Only show legend if not too many actions
                plt.legend(fontsize=12, loc='best', framealpha=0.7)
            else:
                plt.legend(fontsize=14)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to image with higher DPI for better quality
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            # Log to wandb directly through current_metrics
            self.logger.current_metrics['inventory_action_plot'] = wandb.Image(img)
            
            # Close the plot to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error generating inventory-action plot: {e}")

