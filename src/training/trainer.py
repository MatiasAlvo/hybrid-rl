from src import torch, logging, Path, np
# from src.envs.inventory.env import InventoryEnv
# from src.algorithms.hdpo.collectors.collector import InventoryCollector
# from src.algorithms.hdpo.losses.pathwise import HDPOLoss
# from src.envs.base_env import BaseEnvironment
# from src.algorithms.base import BaseAlgorithm
# from src.data.data_handling import Dataset
# from typing import Dict, Optional, Tuple
import os
import copy
import datetime
import matplotlib.pyplot as plt
from src.utils.logger import Logger
# import yaml
import pandas as pd
import wandb
import io
from PIL import Image


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
        self.best_performance_data = {
            'train_loss': np.inf,
            'dev_loss': np.inf,
            'train_loss_reported': np.inf,
            'dev_loss_reported': np.inf,
            'last_epoch_saved': -1000,
            'model_params_to_save': None,
            'optimizer_state_to_save': None,
            'best_epoch': None
        }
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
        ppo_params = optimizer_params.get('ppo_params', {})
        anneal_entropy_coef = ppo_params.get('anneal_entropy_coef', False)
        
        for epoch in range(epochs):
            skip_update_every = optimizer_params.get('skip_update_every_n_epochs', 0)
            skip_update = skip_update_every > 0 and (epoch + 1) % skip_update_every == 0
            if hasattr(optimizer_wrapper, 'set_skip_update'):
                optimizer_wrapper.set_skip_update(skip_update)
            # Update learning rate if annealing is enabled
            if anneal_lr:
                frac = 1.0 - (epoch / epochs)
                for param_group in optimizer_wrapper.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = frac * old_lr
                    # print(f"Group '{param_group.get('name', 'unnamed')}': old_lr={old_lr:.6f}, frac={frac:.3f}, new_lr={new_lr:.6f}")
                    param_group['lr'] = new_lr
                
                # Log LR if logger exists
                if self.logger is not None:
                    self.logger.log_metrics({'train/learning_rate': new_lr}, epoch)
            
            if hasattr(optimizer_wrapper, 'update_entropy_coef'):
                current_entropy_coef = optimizer_wrapper.update_entropy_coef(epoch, epochs)
                if anneal_entropy_coef and self.logger is not None:
                    self.logger.log_metrics({'train/entropy_coef': current_entropy_coef}, epoch)
            
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
                
                # Generate and log plots for dev set with dev loss (if enabled in config)
                if 'trajectory_data' in dev_metrics:
                    logging_params = config.hyperparams_config.get('logging_params', {})
                    normalize_by_mean = logging_params.get('normalize_plots_by_mean_demand', False)
                    
                    # Generate and log inventory vs action plot for dev set with dev loss
                    if logging_params.get('log_inventory_action_plot', False):
                        self.log_inventory_action_plot(
                            dev_metrics['trajectory_data'], 
                            epoch, 
                            dev_loss=dev_metrics['loss/reported'],
                            normalize_by_mean_demand=normalize_by_mean,
                            dump_plot_data=logging_params.get('dump_inventory_action_plot_data', False),
                            dump_plot_data_only=logging_params.get('inventory_action_plot_dump_only', False),
                            dump_plot_data_per_epoch=logging_params.get('inventory_action_plot_dump_per_epoch', True),
                            dump_plot_data_single_file=logging_params.get('inventory_action_plot_dump_single_file', False),
                            dump_plot_data_dir=logging_params.get(
                                'inventory_action_plot_dump_dir',
                                'artifacts/inventory_action_plot_data'
                            ),
                            exp_name=logging_params.get('exp_name', 'default_exp')
                        )
                    
                    # Generate and log inventory vs value plot for dev set with dev loss
                    if logging_params.get('log_inventory_value_plot', False):
                        self.log_inventory_value_plot(
                            dev_metrics['trajectory_data'], 
                            epoch, 
                            dev_loss=dev_metrics['loss/reported'],
                            normalize_by_mean_demand=normalize_by_mean
                        )
                    
                    # Generate and log inventory vs discrete action 0 probability plot for dev set with dev loss
                    if logging_params.get('log_inventory_discrete_action0_prob_plot', False):
                        self.log_inventory_discrete_action0_prob_plot(
                            dev_metrics['trajectory_data'], 
                            epoch, 
                            dev_loss=dev_metrics['loss/reported'],
                            normalize_by_mean_demand=normalize_by_mean,
                            dump_plot_data=logging_params.get('dump_inventory_discrete_action0_prob_plot_data', False),
                            dump_plot_data_only=logging_params.get('inventory_discrete_action0_prob_plot_dump_only', False),
                            dump_plot_data_dir=logging_params.get(
                                'inventory_discrete_action0_prob_plot_dump_dir',
                                'artifacts/inventory_discrete_action0_prob_plot_data'
                            ),
                            run_name=logging_params.get('inventory_discrete_action0_prob_plot_run_name', None),
                            exp_name=logging_params.get('exp_name', 'default_exp')
                        )
                    
                    # Generate and log inventory ordering heatmap (if enabled in config)
                    if logging_params.get('log_inventory_ordering_heatmap', False):
                        self.log_inventory_ordering_heatmap(
                            dev_metrics['trajectory_data'],
                            epoch,
                            dev_loss=dev_metrics['loss/reported']
                        )

                    # Generate and log inventory vs action0 prob heatmap
                    if logging_params.get('log_inventory_action0_heatmap', False):
                        self.log_inventory_action0_heatmap(
                            dev_metrics['trajectory_data'],
                            epoch,
                            dev_loss=dev_metrics['loss/reported'],
                            simulator=simulator,
                            additional_data=additional_data
                        )
                
                self.logger.flush_metrics()
            
            # Update best parameters and save if needed
            self.update_best_params_and_save(
                epoch,
                train_metrics['loss/total'],
                dev_metrics['loss/total'],
                train_metrics['loss/reported'],
                dev_metrics['loss/reported'],
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
        epoch_loss_to_report = 0 # since we ignore the first periods, we don't want to report the loss for the first periods
        total_samples = len(data_loader.dataset)
        periods_tracking_loss = periods - ignore_periods # since we ignore the first periods, we subtract the number of ignored periods from the total number of periods
        
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

        if train and model.trainable and hasattr(optimizer_wrapper, 'on_epoch_start'):
            optimizer_wrapper.on_epoch_start()

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
                # pass trajectory data to optimizer, which computes gradient and updates parameters
                batch_metrics = optimizer_wrapper.optimize(batch_trajectory_data)
                
                # Handle special metrics (like histograms) that shouldn't be averaged
                for k, v in batch_metrics.items():
                    if 'histogram' in k:
                        special_metrics[k] = v  # Store without averaging
                        continue
                        
                    if optimizer_metrics_sum is None:
                        optimizer_metrics_sum = {k: v for k, v in batch_metrics.items() 
                                               if 'histogram' not in k}
                    else:
                        for k, v in batch_metrics.items():
                            if 'histogram' not in k:
                                optimizer_metrics_sum[k] += v
                    num_batches += 1
            
            # Store trajectory data from the first batch only (simplified for now)
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
                    # Skip wandb histogram for now since wandb is disabled
                    # metrics[f'action_distribution/{key}'] = wandb.Histogram(all_data.flatten().cpu().numpy())
                except Exception as e:
                    print(f"Error creating histogram for {key}: {e}")
        
        # Include trajectory data in metrics for visualization
        if not train and trajectory_data is not None:
            metrics['trajectory_data'] = trajectory_data

        if train and model.trainable and hasattr(optimizer_wrapper, 'on_epoch_end'):
            optimizer_wrapper.on_epoch_end()
            if hasattr(optimizer_wrapper, 'get_epoch_metrics'):
                metrics.update(optimizer_wrapper.get_epoch_metrics())

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
            # Add internal data to observation
            observation_and_internal_data = self._prepare_observation_with_internal_data(observation, simulator)

            # Sample action and get policy outputs
            model_output = model(observation_and_internal_data, train=train)
            action_dict = model_output.get('action_dict')
            raw_outputs = model_output.get('raw_outputs', {})
            value = model_output.get('value', None)
            
            # Get vectorized observation from model output (more efficient than recomputing)
            vectorized_obs = model_output.get('vectorized_observation')
            # if vectorized_obs is None:
            #     # Fallback to old method if not available
            #     vectorized_obs = self.vectorize_observation(observation, observation_keys, model)
            
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
            next_vectorized_obs = None
            if vectorized_obs is not None:
                if hasattr(model, 'feature_registry') and model.feature_registry is not None:
                    next_obs_with_internal = self._prepare_observation_with_internal_data(next_observation, simulator)
                    next_vectorized_obs = model.feature_registry.prepare_inputs(next_obs_with_internal, update_ewma=False)
                else:
                    next_vectorized_obs = self.vectorize_observation(next_observation, observation_keys, model)

            # Collect trajectory data if requested
            if collect_trajectories:
                self._collect_trajectory_data(
                    trajectory_data,
                    vectorized_obs,
                    action_dict,
                    value,
                    reward,
                    terminated,
                    raw_outputs,
                    observation=observation,
                    next_vectorized_obs=next_vectorized_obs
                )  # Pass observation

            # Update running rewards
            batch_reward += total_reward
            if t >= ignore_periods:
                reward_to_report += total_reward

            # Update observation
            observation = next_observation

            if terminated:
                break

        # Add final observation to trajectory data
        if collect_trajectories and trajectory_data is not None:
            if trajectory_data.get('next_observations') is not None and len(trajectory_data.get('observations', [])) > 0:
                if hasattr(model, 'feature_registry') and model.feature_registry is not None:
                    final_obs_with_internal = self._prepare_observation_with_internal_data(observation, simulator)
                    final_next_vec = model.feature_registry.prepare_inputs(final_obs_with_internal, update_ewma=False)
                else:
                    final_next_vec = self.vectorize_observation(observation, observation_keys, model)
                trajectory_data['next_observations'].append(final_next_vec.clone())

        # Process collected data
        trajectory_data = self._process_trajectory_data(trajectory_data, collect_trajectories)
        additional_data = self._process_additional_data(additional_data, collect_additional_data)

        # Keep final observation in original dict form for existing advantage bootstrap code.
        if collect_trajectories and trajectory_data is not None:
            trajectory_data['next_observation'] = observation

        return batch_reward, reward_to_report, trajectory_data, additional_data, action_data

    def _get_observation_keys(self, model):
        """Extract observation keys from model if available"""
        if hasattr(model, 'policy') and model.policy is not None:
            return model.policy.observation_keys
        return None
    
    def _validate_observation_keys_consistency(self, model):
        """Validate that policy and value networks use the same observation_keys"""
        # Get policy observation keys
        policy_keys = None
        if hasattr(model, 'policy') and hasattr(model.policy, 'observation_keys'):
            policy_keys = model.policy.observation_keys
        
        # Get value network observation keys
        value_keys = None
        if hasattr(model, 'value_net') and model.value_net is not None and hasattr(model.value_net, 'observation_keys'):
            value_keys = model.value_net.observation_keys
        
        # Only validate if both networks exist and have observation_keys
        if policy_keys is not None and value_keys is not None:
            if policy_keys != value_keys:
                raise NotImplementedError(
                    f"Policy and value networks have different observation_keys. "
                    f"Policy: {policy_keys}, Value: {value_keys}. "
                    f"Logic for different observation_keys is not yet supported."
                )

    def _initialize_trajectory_data(self, collect_trajectories):
        """Initialize trajectory data structure if needed"""
        if not collect_trajectories:
            return None
        
        return {
            'observations': [],  # Keep this for backward compatibility
            'next_observations': [],
            'store_inventories': [],  # NEW: Store raw inventory data
            'discrete_action_indices': [],
            'discrete_logits': [],
            'total_action': [],
            'values': [],
            'rewards': [],
            'terminated': [],
            'raw_continuous_samples': [],
            'past_demands': [],
            'mean_demand': None
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

    def _collect_trajectory_data(self, trajectory_data, vectorized_obs, action_dict, value, reward, terminated, raw_outputs=None, observation=None, next_vectorized_obs=None):
        """Collect trajectory data for the current step"""
        if vectorized_obs is not None:
            if next_vectorized_obs is not None and len(trajectory_data['observations']) > 0:
                trajectory_data['next_observations'].append(vectorized_obs.clone())
            trajectory_data['observations'].append(vectorized_obs.clone())
        
        # NEW: Store raw, unnormalized inventory data
        if observation is not None and 'store_inventories' in observation:
            trajectory_data['store_inventories'].append(observation['store_inventories'].clone())
        
        # NEW: Collect past_demands for normalization
        if observation is not None and 'past_demands' in observation:
            trajectory_data['past_demands'].append(observation['past_demands'].clone())

        # Store per-trajectory mean demand anchor once
        if observation is not None and 'mean_demand' in observation and trajectory_data.get('mean_demand') is None:
            trajectory_data['mean_demand'] = observation['mean_demand'].clone()
        
        # Only append fields that exist and are not None
        if 'discrete_action_indices' in action_dict and action_dict['discrete_action_indices'] is not None:
            trajectory_data['discrete_action_indices'].append(action_dict['discrete_action_indices'].detach().clone())
        
        # Collect discrete logits if available in raw_outputs
        if raw_outputs is not None and 'discrete' in raw_outputs and raw_outputs['discrete'] is not None:
            trajectory_data['discrete_logits'].append(raw_outputs['discrete'].detach().clone())
        
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
            if isinstance(v, torch.Tensor):
                processed_data[k] = v
            elif not v:
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
    
    def update_best_params_and_save(
        self,
        epoch,
        train_loss,
        dev_loss,
        train_loss_reported,
        dev_loss_reported,
        trainer_params,
        model,
        optimizer
    ):
        """
        Update best model parameters if it achieves best performance so far, and save the model
        """
        data_for_compare = {
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'train_loss_reported': train_loss_reported,
            'dev_loss_reported': dev_loss_reported
        }
        if data_for_compare[trainer_params['choose_best_model_on']] < self.best_performance_data[trainer_params['choose_best_model_on']]:  
            self.best_performance_data['train_loss'] = train_loss
            self.best_performance_data['dev_loss'] = dev_loss
            self.best_performance_data['train_loss_reported'] = train_loss_reported
            self.best_performance_data['dev_loss_reported'] = dev_loss_reported
            self.best_performance_data['best_epoch'] = epoch
            if model.trainable:
                # Save the entire model's state dict instead of just the policy
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.state_dict())
                self.best_performance_data['optimizer_state_to_save'] = copy.deepcopy(optimizer.state_dict())
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

    def vectorize_observation(self, observation, observation_keys=None, model=None):
        """
        Convert an observation dictionary into a flat vector based on specified keys.
        
        Parameters:
        -----------
        observation: dict
            The observation dictionary to vectorize
        observation_keys: list, optional
            List of keys from observation to include in vectorization
        model: object, optional
            The model object to validate observation_keys consistency
        """
        # # Validate observation_keys consistency between policy and value networks
        # if not hasattr(self, '_observation_keys_validated') and model is not None:
        #     self._validate_observation_keys_consistency(model)
        #     self._observation_keys_validated = True
        
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

    def log_inventory_action_plot(
        self,
        trajectory_data,
        epoch,
        dev_loss=None,
        normalize_by_mean_demand=False,
        dump_plot_data=False,
        dump_plot_data_only=False,
        dump_plot_data_per_epoch=True,
        dump_plot_data_single_file=False,
        dump_plot_data_dir='artifacts/inventory_action_plot_data',
        exp_name='default_exp'
    ):
        """
        Generate and log a plot showing the relationship between inventory and actions to wandb.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            epoch: Current epoch number
            dev_loss: Optional dev loss to include in the title
            normalize_by_mean_demand: Whether to normalize quantities by mean demand
        """
        try:
            # Skip only when neither wandb logging nor data dumping is requested
            can_log_to_wandb = self.logger is not None and hasattr(self.logger, 'use_wandb') and self.logger.use_wandb
            if not can_log_to_wandb and not dump_plot_data:
                return
            
            # Check if we have the necessary data - use store_inventories instead of observations
            if "store_inventories" not in trajectory_data or "total_action" not in trajectory_data:
                print("Missing required data for inventory-action plot")
                return
            
            # Handle both list and tensor cases for all trajectory data
            if isinstance(trajectory_data["store_inventories"], list):
                inventory_tensor = torch.stack(trajectory_data["store_inventories"], dim=0)
            else:
                inventory_tensor = trajectory_data["store_inventories"]
            
            if isinstance(trajectory_data["total_action"], list):
                action_tensor = torch.stack(trajectory_data["total_action"], dim=0)
            else:
                action_tensor = trajectory_data["total_action"]
            
            # Handle different tensor shapes
            if len(inventory_tensor.shape) >= 3:
                if inventory_tensor.shape[2] > 1: # if we have multiple stores, we need to plot the inventory and action for the first store
                    inventory_tensor = inventory_tensor[:, :, :1]
                    action_tensor = action_tensor[:, :, :1]
                T, B = inventory_tensor.shape[0], inventory_tensor.shape[1]
                # Sum across all remaining dimensions to get total inventory per sample
                if len(inventory_tensor.shape) > 3:
                    # Reshape to [T, B, -1] and sum across the last dimension
                    inventory_tensor = inventory_tensor.reshape(T, B, -1).sum(dim=-1, keepdim=True)
                n_stores = inventory_tensor.shape[2] if len(inventory_tensor.shape) > 2 else 1
            else:
                print(f"ERROR: Unexpected tensor shape: {inventory_tensor.shape}")
                return
            
            # Limit the number of samples to plot
            n_samples = min(30, B)
            
            # Select random batch indices
            random_batch_indices = torch.randperm(B)[:n_samples]
            
            # Select the samples: [T, n_samples, n_stores] and [T, n_samples, 1]
            selected_inventories = inventory_tensor[:, random_batch_indices, :]
            selected_total_action = action_tensor[:, random_batch_indices, :]
            
            # Get discrete action indices if available
            has_discrete_actions = "discrete_action_indices" in trajectory_data and len(trajectory_data["discrete_action_indices"]) > 0
            if has_discrete_actions:
                if isinstance(trajectory_data["discrete_action_indices"], list):
                    discrete_tensor = torch.stack(trajectory_data["discrete_action_indices"], dim=0)
                else:
                    discrete_tensor = trajectory_data["discrete_action_indices"]
                selected_discrete_actions = discrete_tensor[:, random_batch_indices, :]
                discrete_actions_flat = selected_discrete_actions.reshape(-1).detach().cpu().numpy()
            
            # Flatten time and batch dimensions: [T*n_samples, n_stores]
            inventories_flat = selected_inventories.reshape(-1, n_stores).detach().cpu()
            actions_flat = selected_total_action.reshape(-1, 1).detach().cpu()
            
            # Calculate inventory sums (sum across stores)
            inventory_sum = inventories_flat.sum(dim=1).numpy()
            total_action = actions_flat.squeeze().numpy()
            
            # Apply normalization if requested
            if normalize_by_mean_demand and 'past_demands' in trajectory_data and len(trajectory_data['past_demands']) > 0:
                # Handle both list and tensor cases for past_demands
                if isinstance(trajectory_data['past_demands'], list):
                    past_demands_tensor = torch.stack(trajectory_data['past_demands'], dim=0)
                else:
                    past_demands_tensor = trajectory_data['past_demands']
                
                # Select the same samples: [T, n_samples, n_stores, n_periods]
                selected_past_demands = past_demands_tensor[:, random_batch_indices, :, :]
                
                # Compute mean demand per sample per timestep: [T, n_samples]
                mean_demands = selected_past_demands.mean(dim=(2, 3))  # Mean across stores and periods
                                
                # Flatten: [T * n_samples]
                mean_demands_flat = mean_demands.reshape(-1).cpu().numpy()
                
                # Normalize inventory and actions
                inventory_sum = inventory_sum / (mean_demands_flat + 1e-8)
                total_action = total_action / (mean_demands_flat + 1e-8)

            # Optional one-off dump for offline plotting/debugging.
            if dump_plot_data:
                output_dir = os.path.join(dump_plot_data_dir, exp_name)
                os.makedirs(output_dir, exist_ok=True)

                if dump_plot_data_per_epoch:
                    file_path = os.path.join(
                        output_dir,
                        f"inventory_action_epoch_{epoch:04d}_ts_{self.time_stamp}.npz"
                    )
                    dump_payload = {
                        'inventory_sum': inventory_sum,
                        'total_action': total_action,
                        'epoch': np.array([epoch]),
                        'normalize_by_mean_demand': np.array([int(normalize_by_mean_demand)]),
                    }
                    if has_discrete_actions:
                        dump_payload['discrete_actions'] = discrete_actions_flat
                    if dev_loss is not None:
                        dump_payload['dev_loss'] = np.array([float(dev_loss)])
                    np.savez_compressed(file_path, **dump_payload)

                if dump_plot_data_single_file:
                    aggregate_file_path = os.path.join(
                        output_dir,
                        f"inventory_action_all_epochs_ts_{self.time_stamp}.npz"
                    )

                    current_inventory = np.asarray(inventory_sum, dtype=np.float32).reshape(1, -1)
                    current_action = np.asarray(total_action, dtype=np.float32).reshape(1, -1)
                    if has_discrete_actions:
                        current_discrete = np.asarray(discrete_actions_flat, dtype=np.float32).reshape(1, -1)
                    else:
                        current_discrete = np.full_like(current_inventory, np.nan, dtype=np.float32)
                    current_num_points = np.array([current_inventory.shape[1]], dtype=np.int32)
                    current_epoch = np.array([epoch], dtype=np.int32)
                    current_dev_loss = np.array(
                        [float(dev_loss) if dev_loss is not None else np.nan],
                        dtype=np.float32
                    )
                    current_normalized = np.array([int(normalize_by_mean_demand)], dtype=np.int32)

                    if os.path.exists(aggregate_file_path):
                        with np.load(aggregate_file_path) as existing:
                            inv_existing = existing['inventory_sum_by_epoch']
                            act_existing = existing['total_action_by_epoch']
                            disc_existing = existing['discrete_actions_by_epoch']
                            epoch_existing = existing['epoch_by_row']
                            points_existing = existing['num_points_by_epoch']
                            dev_loss_existing = existing['dev_loss_reported_by_epoch']
                            normalized_existing = existing['normalize_by_mean_demand_by_epoch']
                    else:
                        inv_existing = np.empty((0, 0), dtype=np.float32)
                        act_existing = np.empty((0, 0), dtype=np.float32)
                        disc_existing = np.empty((0, 0), dtype=np.float32)
                        epoch_existing = np.empty((0,), dtype=np.int32)
                        points_existing = np.empty((0,), dtype=np.int32)
                        dev_loss_existing = np.empty((0,), dtype=np.float32)
                        normalized_existing = np.empty((0,), dtype=np.int32)

                    max_points = max(
                        inv_existing.shape[1] if inv_existing.ndim == 2 and inv_existing.size > 0 else 0,
                        current_inventory.shape[1]
                    )

                    def _pad_to_width(array_2d, width):
                        if array_2d.size == 0:
                            return np.full((0, width), np.nan, dtype=np.float32)
                        if array_2d.shape[1] == width:
                            return array_2d
                        pad_width = width - array_2d.shape[1]
                        return np.pad(
                            array_2d,
                            ((0, 0), (0, pad_width)),
                            mode='constant',
                            constant_values=np.nan
                        )

                    inv_existing = _pad_to_width(inv_existing, max_points)
                    act_existing = _pad_to_width(act_existing, max_points)
                    disc_existing = _pad_to_width(disc_existing, max_points)
                    current_inventory = _pad_to_width(current_inventory, max_points)
                    current_action = _pad_to_width(current_action, max_points)
                    current_discrete = _pad_to_width(current_discrete, max_points)

                    np.savez_compressed(
                        aggregate_file_path,
                        inventory_sum_by_epoch=np.vstack([inv_existing, current_inventory]),
                        total_action_by_epoch=np.vstack([act_existing, current_action]),
                        discrete_actions_by_epoch=np.vstack([disc_existing, current_discrete]),
                        epoch_by_row=np.concatenate([epoch_existing, current_epoch]),
                        num_points_by_epoch=np.concatenate([points_existing, current_num_points]),
                        dev_loss_reported_by_epoch=np.concatenate([dev_loss_existing, current_dev_loss]),
                        normalize_by_mean_demand_by_epoch=np.concatenate([normalized_existing, current_normalized]),
                    )

            if dump_plot_data_only:
                return
            
            # Create the plot
            plt.figure(figsize=(16, 10))
            
            # Color points by discrete action if available
            if has_discrete_actions:
                unique_actions = np.unique(discrete_actions_flat)
                
                if len(unique_actions) <= 10:
                    cmap = plt.colormaps['tab10']
                else:
                    cmap = plt.colormaps['hsv']
                
                for i, action in enumerate(unique_actions):
                    mask = np.where(discrete_actions_flat == action)[0]
                    plt.scatter(
                        inventory_sum[mask], 
                        total_action[mask], 
                        alpha=0.8, 
                        s=30,
                        color=cmap(i % cmap.N),
                        label=f'Action {int(action)}'
                    )
            else:
                plt.scatter(inventory_sum, total_action, alpha=0.7, s=50)
            
            # Only add (s, S) policy line if NOT normalizing
            if not normalize_by_mean_demand:
                s, S = 26, 62  # Example values
                inventory_range = np.linspace(np.min(inventory_sum), np.max(inventory_sum), 100)
                order_amounts = np.maximum(S - inventory_range, 0) * (inventory_range <= s)
                plt.plot(inventory_range, order_amounts, color='black', linewidth=3, label='(s, S) Policy')
            
            # Add title with normalization indicator
            title = f'Inventory vs Actions (Epoch {epoch})'
            if normalize_by_mean_demand:
                title += ' [Normalized by Mean Demand]'
            if dev_loss is not None:
                title += f' - Dev Loss: {dev_loss:.4f}'
            
            plt.title(title, fontsize=18)
            xlabel = 'Total Inventory (Normalized)' if normalize_by_mean_demand else 'Total Inventory'
            ylabel = 'Total Actions (Normalized)' if normalize_by_mean_demand else 'Total Actions'
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            if has_discrete_actions and len(unique_actions) <= 10:
                plt.legend(fontsize=12, loc='best', framealpha=0.7)
            elif not normalize_by_mean_demand:  # Only show legend for (s,S) if not normalized
                plt.legend(fontsize=14)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            if can_log_to_wandb:
                self.logger.current_metrics['inventory_action_plot'] = wandb.Image(img)
            plt.close()
            
        except Exception as e:
            print(f"Error generating inventory-action plot: {e}")
            import traceback
            traceback.print_exc()

    def log_inventory_value_plot(self, trajectory_data, epoch, dev_loss=None, normalize_by_mean_demand=False):
        """
        Generate and log a plot showing the relationship between inventory and value agent outputs to wandb.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            epoch: Current epoch number
            dev_loss: Optional dev loss to include in the title
            normalize_by_mean_demand: Whether to normalize quantities by mean demand
        """
        try:
            # Skip if logger is not available
            if self.logger is None or not hasattr(self.logger, 'use_wandb') or not self.logger.use_wandb:
                return
            
            # Check if we have the necessary data - use store_inventories instead of observations
            if "store_inventories" not in trajectory_data or "values" not in trajectory_data:
                print("Missing required data for inventory-value plot")
                return
            
            # Handle both list and tensor cases for all trajectory data
            if isinstance(trajectory_data["store_inventories"], list):
                inventory_tensor = torch.stack(trajectory_data["store_inventories"], dim=0)
            else:
                inventory_tensor = trajectory_data["store_inventories"]
            
            if isinstance(trajectory_data["values"], list):
                value_tensor = torch.stack(trajectory_data["values"], dim=0)
            else:
                value_tensor = trajectory_data["values"]
            
            # Handle different tensor shapes
            if len(inventory_tensor.shape) >= 3:
                if inventory_tensor.shape[2] > 1:
                    raise NotImplementedError("Plot for multi-store is not implemented")
                T, B = inventory_tensor.shape[0], inventory_tensor.shape[1]
                # Sum across all remaining dimensions to get total inventory per sample
                if len(inventory_tensor.shape) > 3:
                    # Reshape to [T, B, -1] and sum across the last dimension
                    inventory_tensor = inventory_tensor.reshape(T, B, -1).sum(dim=-1, keepdim=True)
                n_stores = inventory_tensor.shape[2] if len(inventory_tensor.shape) > 2 else 1
            else:
                print(f"ERROR: Unexpected tensor shape: {inventory_tensor.shape}")
                return
            
            # Limit the number of samples to plot
            n_samples = min(30, B)
            
            # Select random batch indices
            random_batch_indices = torch.randperm(B)[:n_samples]
            
            # Select the samples: [T, n_samples, n_stores] and [T, n_samples, 1]
            selected_inventories = inventory_tensor[:, random_batch_indices, :]
            selected_values = value_tensor[:, random_batch_indices, :]
            
            # Get discrete action indices if available for coloring
            has_discrete_actions = "discrete_action_indices" in trajectory_data
            if has_discrete_actions:
                if isinstance(trajectory_data["discrete_action_indices"], list):
                    discrete_tensor = torch.stack(trajectory_data["discrete_action_indices"], dim=0)
                else:
                    discrete_tensor = trajectory_data["discrete_action_indices"]
                selected_discrete_actions = discrete_tensor[:, random_batch_indices, :]  # Shape: [T, n_samples, 1]
                # Flatten and convert to numpy
                discrete_actions_flat = selected_discrete_actions.reshape(-1).detach().cpu().numpy()
            
            # Flatten time and batch dimensions: [T*n_samples, n_stores]
            inventories_flat = selected_inventories.reshape(-1, n_stores).detach().cpu()
            values_flat = selected_values.reshape(-1, 1).detach().cpu()
            
            # Calculate inventory sums (sum across stores)
            inventory_sum = inventories_flat.sum(dim=1).numpy()
            value_outputs = values_flat.squeeze().numpy()
            
            # Apply normalization if requested
            if normalize_by_mean_demand and 'past_demands' in trajectory_data and len(trajectory_data['past_demands']) > 0:
                # Handle both list and tensor cases for past_demands
                if isinstance(trajectory_data['past_demands'], list):
                    past_demands_tensor = torch.stack(trajectory_data['past_demands'], dim=0)
                else:
                    past_demands_tensor = trajectory_data['past_demands']
                
                # Select the same samples: [T, n_samples, n_stores, n_periods]
                selected_past_demands = past_demands_tensor[:, random_batch_indices, :, :]
                
                # Compute mean demand per sample per timestep: [T, n_samples]
                mean_demands = selected_past_demands.mean(dim=(2, 3))  # Mean across stores and periods
                
                # Clamp to stabilize (same as prepare_inputs logic)
                mean_demands = torch.clamp(mean_demands, min=1.0, max=100.0)
                
                # Flatten: [T * n_samples]
                mean_demands_flat = mean_demands.reshape(-1).cpu().numpy()
                
                # Normalize inventory and actions
                inventory_sum = inventory_sum / (mean_demands_flat + 1e-8)
                value_outputs = value_outputs / (mean_demands_flat + 1e-8)
            
            # Create the plot with larger figure size
            plt.figure(figsize=(16, 10))
            
            # Color points by discrete action if available
            if has_discrete_actions:
                # Get unique discrete actions for coloring
                unique_actions = np.unique(discrete_actions_flat)
                
                # Create a colormap with distinct colors
                if len(unique_actions) <= 10:
                    # For few actions, use tab10 colormap
                    cmap = plt.colormaps['tab10']
                else:
                    # For many actions, use hsv colormap
                    cmap = plt.colormaps['hsv']
                
                # Plot each discrete action with a different color
                for i, action in enumerate(unique_actions):
                    # Fix boolean mask conversion warning by explicitly converting to integer indices
                    mask = np.where(discrete_actions_flat == action)[0]
                    plt.scatter(
                        inventory_sum[mask], 
                        value_outputs[mask], 
                        alpha=0.8, 
                        s=30,  # Larger points for better visibility
                        color=cmap(i % cmap.N),  # Use modulo to ensure we don't exceed colormap range
                        label=f'Action {int(action)}'
                    )
            else:
                # If no discrete actions, use a single color
                plt.scatter(inventory_sum, value_outputs, alpha=0.7, s=50)
            
            # Add dev loss to title if provided
            title = f'Inventory vs Value Agent Outputs (Epoch {epoch})'
            if normalize_by_mean_demand:
                title += ' [Inventory Normalized by Mean Demand]'
            if dev_loss is not None:
                title += f' - Dev Loss: {dev_loss:.4f}'
            
            # Use larger font sizes for better readability
            plt.title(title, fontsize=18)
            plt.xlabel('Total Inventory (Normalized)' if normalize_by_mean_demand else 'Total Inventory', fontsize=16)
            plt.ylabel('Value Agent Output', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            # Add legend with reasonable size and position
            if has_discrete_actions and len(unique_actions) <= 10:  # Only show legend if not too many actions
                plt.legend(fontsize=12, loc='best', framealpha=0.7)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to image with higher DPI for better quality
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            # Log to wandb directly through current_metrics
            self.logger.current_metrics['inventory_value_plot'] = wandb.Image(img)
            
            # Close the plot to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error generating inventory-value plot: {e}")

    def log_inventory_discrete_action0_prob_plot(
        self,
        trajectory_data,
        epoch,
        dev_loss=None,
        normalize_by_mean_demand=False,
        dump_plot_data=False,
        dump_plot_data_only=False,
        dump_plot_data_dir='artifacts/inventory_discrete_action0_prob_plot_data',
        run_name=None,
        exp_name='default_exp'
    ):
        """
        Generate and log a plot showing the relationship between inventory and probability of discrete action 0.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            epoch: Current epoch number
            dev_loss: Optional dev loss to include in the title
            normalize_by_mean_demand: Whether to normalize quantities by mean demand
        """
        try:
            can_log_to_wandb = self.logger is not None and hasattr(self.logger, 'use_wandb') and self.logger.use_wandb
            if not can_log_to_wandb and not dump_plot_data:
                return
            
            # Check if we have the necessary data - use store_inventories instead of observations
            if ("store_inventories" not in trajectory_data or 
                "discrete_logits" not in trajectory_data or
                len(trajectory_data["discrete_logits"]) == 0):
                print("Missing required data for inventory-discrete action 0 probability plot")
                return
            
            # Handle both list and tensor cases for all trajectory data
            if isinstance(trajectory_data["store_inventories"], list):
                inventory_tensor = torch.stack(trajectory_data["store_inventories"], dim=0)
            else:
                inventory_tensor = trajectory_data["store_inventories"]
            
            if isinstance(trajectory_data["discrete_logits"], list):
                discrete_logits_tensor = torch.stack(trajectory_data["discrete_logits"], dim=0)
            else:
                discrete_logits_tensor = trajectory_data["discrete_logits"]
            
            # Handle different tensor shapes
            if len(inventory_tensor.shape) >= 3:
                T, B = inventory_tensor.shape[0], inventory_tensor.shape[1]
                # Sum across all remaining dimensions to get total inventory per sample
                if len(inventory_tensor.shape) > 3:
                    # Reshape to [T, B, -1] and sum across the last dimension
                    inventory_tensor = inventory_tensor.reshape(T, B, -1).sum(dim=-1, keepdim=True)
                n_stores = inventory_tensor.shape[2] if len(inventory_tensor.shape) > 2 else 1
            else:
                print(f"ERROR: Unexpected tensor shape: {inventory_tensor.shape}")
                return
            
            # Limit the number of samples to plot
            n_samples = min(30, B)
            
            # Select random batch indices
            random_batch_indices = torch.randperm(B)[:n_samples]
            
            # Select the samples: [T, n_samples, n_stores] and [T, n_samples, n_actions]
            selected_inventories = inventory_tensor[:, random_batch_indices, :]
            selected_discrete_logits = discrete_logits_tensor[:, random_batch_indices, :]
            
            # Flatten time and batch dimensions: [T*n_samples, n_stores] and [T*n_samples, n_actions]
            inventories_flat = selected_inventories.reshape(-1, n_stores).detach().cpu()
            discrete_logits_flat = selected_discrete_logits.reshape(-1, selected_discrete_logits.shape[-1]).detach().cpu()
            
            # Calculate inventory sum (sum across stores)
            inventory_sum = inventories_flat.sum(dim=1).numpy()
            
            # Apply normalization if requested
            if normalize_by_mean_demand and 'past_demands' in trajectory_data and len(trajectory_data['past_demands']) > 0:
                # Handle both list and tensor cases for past_demands
                if isinstance(trajectory_data['past_demands'], list):
                    past_demands_tensor = torch.stack(trajectory_data['past_demands'], dim=0)
                else:
                    past_demands_tensor = trajectory_data['past_demands']
                
                # Select the same samples: [T, n_samples, n_stores, n_periods]
                selected_past_demands = past_demands_tensor[:, random_batch_indices, :, :]
                
                # Compute mean demand per sample per timestep: [T, n_samples]
                mean_demands = selected_past_demands.mean(dim=(2, 3))  # Mean across stores and periods
                
                # Clamp to stabilize (same as prepare_inputs logic)
                mean_demands = torch.clamp(mean_demands, min=1.0, max=100.0)
                
                # Flatten: [T * n_samples]
                mean_demands_flat = mean_demands.reshape(-1).cpu().numpy()
                
                # Normalize inventory
                inventory_sum = inventory_sum / (mean_demands_flat + 1e-8)
        
            # Compute probabilities from logits
            discrete_probs = torch.softmax(discrete_logits_flat, dim=-1)
            action0_probs = discrete_probs[:, 0].numpy()

            # Optional one-off dump for offline plotting/debugging.
            if dump_plot_data:
                resolved_run_name = run_name if run_name else f"run_{self.time_stamp}"
                safe_run_name = str(resolved_run_name).replace('/', '_').replace(' ', '_')
                output_dir = os.path.join(dump_plot_data_dir, exp_name, safe_run_name)
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(
                    output_dir,
                    f"inventory_discrete_action0_prob_epoch_{epoch:04d}_ts_{self.time_stamp}.npz"
                )
                dump_payload = {
                    'inventory_sum': inventory_sum,
                    'action0_prob': action0_probs,
                    'epoch': np.array([epoch]),
                    'normalize_by_mean_demand': np.array([int(normalize_by_mean_demand)]),
                }
                if dev_loss is not None:
                    dump_payload['dev_loss_reported'] = np.array([float(dev_loss)])
                dump_payload['run_name'] = np.array([safe_run_name])
                np.savez_compressed(file_path, **dump_payload)

            if dump_plot_data_only:
                return
            
            # Create the plot
            plt.figure(figsize=(16, 10))
            plt.scatter(inventory_sum, action0_probs, alpha=0.7, s=50, color='blue')
            
            # Add title with normalization indicator
            title = f'Inventory vs Probability of Discrete Action 0 (Epoch {epoch})'
            if normalize_by_mean_demand:
                title += ' [Inventory Normalized by Mean Demand]'
            if dev_loss is not None:
                title += f' - Dev Loss: {dev_loss:.4f}'
            
            plt.title(title, fontsize=18)
            xlabel = 'Total Inventory (Normalized)' if normalize_by_mean_demand else 'Total Inventory'
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel('Probability of Discrete Action 0', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylim(0, 1)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            if can_log_to_wandb:
                self.logger.current_metrics['inventory_discrete_action0_prob_plot'] = wandb.Image(img)
            plt.close()
            
        except Exception as e:
            print(f"Error generating inventory-discrete action 0 probability plot: {e}")
            import traceback
            traceback.print_exc()
            
    def log_inventory_ordering_heatmap(self, trajectory_data, epoch, dev_loss=None, store_0=0, store_1=1):
        """
        Generate and log a heatmap showing the probability of ordering as a function of 
        inventory levels at two stores.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            epoch: Current epoch number
            dev_loss: Optional dev loss to include in the title
            store_0: Index of first store for x-axis (default: 0)
            store_1: Index of second store for y-axis (default: 1)
        """
        try:
            if self.logger is None or not hasattr(self.logger, 'use_wandb') or not self.logger.use_wandb:
                return
            
            # Check if we have the necessary data
            if ("store_inventories" not in trajectory_data or 
                "discrete_logits" not in trajectory_data or
                len(trajectory_data["discrete_logits"]) == 0):
                print("Missing required data for inventory ordering heatmap")
                return
            
            # Handle both list and tensor cases
            if isinstance(trajectory_data["store_inventories"], list):
                inventory_tensor = torch.stack(trajectory_data["store_inventories"], dim=0)
            else:
                inventory_tensor = trajectory_data["store_inventories"]
            
            if isinstance(trajectory_data["discrete_logits"], list):
                discrete_logits_tensor = torch.stack(trajectory_data["discrete_logits"], dim=0)
            else:
                discrete_logits_tensor = trajectory_data["discrete_logits"]
            
            print(f"DEBUG: inventory_tensor.shape = {inventory_tensor.shape}")
            
            # Handle different tensor shapes - need original multi-store data
            if len(inventory_tensor.shape) >= 3:
                T, B = inventory_tensor.shape[0], inventory_tensor.shape[1]
                # If it's already been summed/reduced, we need the original data
                if inventory_tensor.shape[2] == 1:
                    print("Inventory has already been summed across stores. Cannot create 2D heatmap.")
                    print("This plot requires per-store inventory data (shape should be [T, B, n_stores] with n_stores >= 2)")
                    return
                n_stores = inventory_tensor.shape[2]
            else:
                print(f"ERROR: Unexpected tensor shape: {inventory_tensor.shape}")
                return
            
            # Check if we have enough stores
            if n_stores < 2:
                print(f"Need at least 2 stores for 2D heatmap, but only have {n_stores}")
                return
            
            if store_0 >= n_stores or store_1 >= n_stores:
                print(f"Store indices ({store_0}, {store_1}) out of range for {n_stores} stores")
                return
            
            # Flatten time and batch: [T*B, n_stores] and [T*B, n_actions]
            inventories_flat = inventory_tensor.reshape(-1, n_stores).detach().cpu()
            discrete_logits_flat = discrete_logits_tensor.reshape(-1, discrete_logits_tensor.shape[-1]).detach().cpu()
            
            # Extract inventory for the two stores and round to nearest integer
            inv_store_0 = torch.round(inventories_flat[:, store_0]).numpy().astype(int)
            inv_store_1 = torch.round(inventories_flat[:, store_1]).numpy().astype(int)
            
            # Compute probabilities and extract ordering probability (action index 1)
            discrete_probs = torch.softmax(discrete_logits_flat, dim=-1)
            ordering_probs = discrete_probs[:, 1].numpy()
            
            # Create bins for the heatmap
            min_inv_0, max_inv_0 = inv_store_0.min(), inv_store_0.max()
            min_inv_1, max_inv_1 = inv_store_1.min(), inv_store_1.max()
            
            # Create grid
            x_bins = np.arange(min_inv_0, max_inv_0 + 2)  # +2 to include max
            y_bins = np.arange(min_inv_1, max_inv_1 + 2)
            
            # Initialize arrays to accumulate probabilities and counts
            prob_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
            counts = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
            
            # Accumulate probabilities for each bucket
            for i in range(len(inv_store_0)):
                x_idx = np.searchsorted(x_bins, inv_store_0[i], side='right') - 1
                y_idx = np.searchsorted(y_bins, inv_store_1[i], side='right') - 1
                
                # Ensure indices are within bounds
                if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                    prob_sum[y_idx, x_idx] += ordering_probs[i]
                    counts[y_idx, x_idx] += 1
            
            # Compute average probability for each bucket
            avg_probs = np.divide(prob_sum, counts, where=counts > 0, out=np.full_like(prob_sum, np.nan))
            
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Use imshow for heatmap
            im = ax.imshow(avg_probs, origin='lower', aspect='auto', 
                        extent=[min_inv_0, max_inv_0 + 1, min_inv_1, max_inv_1 + 1],
                        cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability of Ordering (Action 1)', fontsize=14)
            
            # Set labels and title
            title = f'Ordering Probability Heatmap (Epoch {epoch})'
            if dev_loss is not None:
                title += f' - Dev Loss: {dev_loss:.4f}'
            ax.set_title(title, fontsize=18)
            ax.set_xlabel(f'Inventory at Store {store_0} (x₁)', fontsize=16)
            ax.set_ylabel(f'Inventory at Store {store_1} (x₂)', fontsize=16)
            
            # Add grid
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = Image.open(buf)
            
            self.logger.current_metrics['inventory_ordering_heatmap'] = wandb.Image(img)
            plt.close()
            
            
        except Exception as e:
            print(f"Error generating inventory ordering heatmap: {e}")
            import traceback
            traceback.print_exc()

    def log_inventory_action0_heatmap(self, trajectory_data, epoch, dev_loss=None, store_0=0, store_1=1, simulator=None, additional_data=None):
        """
        Generate and log a heatmap showing the probability of discrete action 0
        as a function of inventory levels at two stores.
        """
        try:
            if self.logger is None or not hasattr(self.logger, 'use_wandb') or not self.logger.use_wandb:
                return
            
            if ("store_inventories" not in trajectory_data or 
                "discrete_logits" not in trajectory_data or
                len(trajectory_data["discrete_logits"]) == 0):
                print("Missing required data for inventory action0 heatmap")
                return
            
            if isinstance(trajectory_data["store_inventories"], list):
                inventory_tensor = torch.stack(trajectory_data["store_inventories"], dim=0)
            else:
                inventory_tensor = trajectory_data["store_inventories"]
            
            if isinstance(trajectory_data["discrete_logits"], list):
                discrete_logits_tensor = torch.stack(trajectory_data["discrete_logits"], dim=0)
            else:
                discrete_logits_tensor = trajectory_data["discrete_logits"]
            
            if len(inventory_tensor.shape) >= 3:
                T, B = inventory_tensor.shape[0], inventory_tensor.shape[1]
                if inventory_tensor.shape[2] == 1:
                    print("Inventory has already been summed across stores; need per-store inventories.")
                    return
                n_stores = inventory_tensor.shape[2]
            else:
                print(f"ERROR: Unexpected tensor shape: {inventory_tensor.shape}")
                return
            
            if n_stores < 2:
                print(f"Need at least 2 stores for 2D heatmap, but only have {n_stores}")
                return
            
            if store_0 >= n_stores or store_1 >= n_stores:
                print(f"Store indices ({store_0}, {store_1}) out of range for {n_stores} stores")
                return
            
            inventories_flat = inventory_tensor.reshape(-1, n_stores, *inventory_tensor.shape[3:]).detach().cpu()
            discrete_logits_flat = discrete_logits_tensor.reshape(-1, discrete_logits_tensor.shape[-1]).detach().cpu()
            
            # Use on-hand inventory if lead-time dimension exists
            if inventories_flat.dim() == 3:
                inventories_flat = inventories_flat[:, :, 0]
            
            inv_store_0 = torch.round(inventories_flat[:, store_0]).numpy().astype(int)
            inv_store_1 = torch.round(inventories_flat[:, store_1]).numpy().astype(int)
            
            discrete_probs = torch.softmax(discrete_logits_flat, dim=-1)
            action0_probs = discrete_probs[:, 0].numpy()
            
            min_inv_0, max_inv_0 = inv_store_0.min(), inv_store_0.max()
            min_inv_1, max_inv_1 = inv_store_1.min(), inv_store_1.max()
            
            x_bins = np.arange(min_inv_0, max_inv_0 + 2)
            y_bins = np.arange(min_inv_1, max_inv_1 + 2)
            
            prob_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
            counts = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
            
            for i in range(len(inv_store_0)):
                x_idx = np.searchsorted(x_bins, inv_store_0[i], side='right') - 1
                y_idx = np.searchsorted(y_bins, inv_store_1[i], side='right') - 1
                if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                    prob_sum[y_idx, x_idx] += action0_probs[i]
                    counts[y_idx, x_idx] += 1
            
            avg_probs = np.divide(prob_sum, counts, where=counts > 0, out=np.full_like(prob_sum, np.nan))

            overlay_coords = None
            if simulator is not None and hasattr(simulator, 'problem_params'):
                lqr_params = simulator.problem_params.get('lqr', {})
                if all(k in lqr_params for k in ('A', 'B', 'Q', 'R')):
                    A = torch.tensor(lqr_params['A'], dtype=torch.float32)
                    B = torch.tensor(lqr_params['B'], dtype=torch.float32)
                    Q = torch.tensor(lqr_params['Q'], dtype=torch.float32)
                    R = torch.tensor(lqr_params['R'], dtype=torch.float32)

                    default_P = [
                        [[6.065, 1.206], [1.206, 1.905]],
                        [[9.087, 3.235], [3.235, 2.348]],
                        [[5.108, 1.266], [1.266, 1.935]],
                        [[7.219, 2.561], [2.561, 2.107]],
                    ]
                    P_list = simulator.problem_params.get('lqr_policy_P_matrices', default_P)
                    P = torch.tensor(P_list, dtype=torch.float32)

                    n_modes = A.shape[0]
                    n_p = P.shape[0]
                    rho_list = []
                    eye = torch.eye(B.shape[2], dtype=torch.float32)
                    for mode_idx in range(n_modes):
                        rho_mode = []
                        for p_idx in range(n_p):
                            BtP = B[mode_idx].transpose(0, 1) @ P[p_idx]
                            inv_term = torch.linalg.inv(R[mode_idx] + BtP @ B[mode_idx] + 1e-8 * eye)
                            rho = (
                                Q[mode_idx]
                                + A[mode_idx].transpose(0, 1) @ P[p_idx] @ A[mode_idx]
                                - A[mode_idx].transpose(0, 1) @ P[p_idx] @ B[mode_idx] @ inv_term @ BtP @ A[mode_idx]
                            )
                            rho_mode.append(rho)
                        rho_list.append(torch.stack(rho_mode, dim=0))
                    rho_mats = torch.stack(rho_list, dim=0)  # [m, p, 2, 2]

                    x_centers = (x_bins[:-1] + x_bins[1:]) / 2.0
                    y_centers = (y_bins[:-1] + y_bins[1:]) / 2.0
                    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
                    points = torch.tensor(
                        np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1),
                        dtype=torch.float32
                    )  # [N, 2]
                    scores = torch.einsum('ni,mkij,nj->nmk', points, rho_mats, points)
                    flat_scores = scores.reshape(points.shape[0], -1)
                    best_flat_idx = torch.argmin(flat_scores, dim=-1)
                    best_mode = (best_flat_idx // n_p)
                    action0_mask = (best_mode == 0).reshape(grid_x.shape)
                    overlay_coords = (grid_x[action0_mask], grid_y[action0_mask])

            def log_heatmap(values, metric_key, colorbar_label, title_prefix, vmin=None, vmax=None):
                fig, ax = plt.subplots(figsize=(14, 10))
                im = ax.imshow(
                    values,
                    origin='lower',
                    aspect='auto',
                    extent=[min_inv_0, max_inv_0 + 1, min_inv_1, max_inv_1 + 1],
                    cmap='RdYlBu_r',
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest'
                )

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(colorbar_label, fontsize=14)

                title = f'{title_prefix} (Epoch {epoch})'
                if dev_loss is not None:
                    title += f' - Dev Loss: {dev_loss:.4f}'
                ax.set_title(title, fontsize=18)
                ax.set_xlabel('s₁', fontsize=16)
                ax.set_ylabel('s₂', fontsize=16)
                ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
                if overlay_coords is not None and overlay_coords[0].size > 0:
                    ax.scatter(
                        overlay_coords[0],
                        overlay_coords[1],
                        marker='*',
                        s=20,
                        color='black',
                        alpha=0.7
                    )

                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                img = Image.open(buf)

                self.logger.current_metrics[metric_key] = wandb.Image(img)
                plt.close()

            log_heatmap(
                avg_probs,
                'inventory_action0_heatmap',
                'Probability of Mode 1',
                'Mode 1 Probability Heatmap',
                vmin=0,
                vmax=1
            )

            continuous_values = None
            if additional_data is not None and isinstance(additional_data, dict):
                continuous_values = additional_data.get('continuous_values')

            discrete_indices_tensor = trajectory_data.get('discrete_action_indices')

            if continuous_values is not None and discrete_indices_tensor is not None:
                if isinstance(continuous_values, list):
                    continuous_values_tensor = torch.stack(continuous_values, dim=0)
                else:
                    continuous_values_tensor = continuous_values

                if isinstance(discrete_indices_tensor, list):
                    discrete_indices_tensor = torch.stack(discrete_indices_tensor, dim=0)

                continuous_values_flat = continuous_values_tensor.reshape(
                    -1,
                    continuous_values_tensor.shape[2],
                    continuous_values_tensor.shape[-1]
                ).detach().cpu()

                discrete_indices_flat = discrete_indices_tensor.reshape(-1, *discrete_indices_tensor.shape[2:]).detach().cpu()

                if discrete_indices_flat.dim() == 1:
                    selected_modes = discrete_indices_flat
                elif discrete_indices_flat.dim() == 2:
                    if discrete_indices_flat.shape[1] == 1:
                        selected_modes = discrete_indices_flat[:, 0]
                    else:
                        selected_modes = discrete_indices_flat[:, store_0]
                else:
                    print(f"Unexpected discrete action indices shape: {tuple(discrete_indices_flat.shape)}")
                    selected_modes = None

                if selected_modes is not None:
                    if store_0 >= continuous_values_flat.shape[1]:
                        print(f"Store index {store_0} out of range for continuous values.")
                        return

                    selected_modes = selected_modes.long().clamp(min=0, max=continuous_values_flat.shape[-1] - 1)
                    action0_values = continuous_values_flat[
                        torch.arange(continuous_values_flat.shape[0]),
                        store_0,
                        selected_modes
                    ].numpy()

                    action_sum = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
                    action_counts = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

                    for i in range(len(inv_store_0)):
                        x_idx = np.searchsorted(x_bins, inv_store_0[i], side='right') - 1
                        y_idx = np.searchsorted(y_bins, inv_store_1[i], side='right') - 1
                        if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                            action_sum[y_idx, x_idx] += action0_values[i]
                            action_counts[y_idx, x_idx] += 1

                    avg_actions = np.divide(
                        action_sum,
                        action_counts,
                        where=action_counts > 0,
                        out=np.full_like(action_sum, np.nan)
                    )

                    log_heatmap(
                        avg_actions,
                        'inventory_action0_continuous_heatmap',
                        f'Continuous Action (Store {store_0}, Selected Mode)',
                        f'Selected Continuous Action Heatmap'
                    )
            else:
                print("Missing continuous values or discrete action indices for continuous action heatmap.")
        except Exception as e:
            print(f"Error generating inventory action0 heatmap: {e}")
            import traceback
            traceback.print_exc()
