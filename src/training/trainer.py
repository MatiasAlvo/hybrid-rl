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
        
        # Initialize logger with complete config
        logger = Logger(config, model)
        global_step = 0
        
        final_metrics = {}  # Store the final metrics
        
        for epoch in range(epochs):
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
            logger.watch_model()  # This will only take effect once
            
            # Validation epoch
            with torch.no_grad():
                dev_metrics = self.do_one_epoch(
                    optimizer_wrapper,
                    data_loaders['dev'],
                    loss_function,
                    simulator,
                    model,
                    params_by_dataset['dev']['periods'],
                    problem_params,
                    observation_params,
                    train=False,
                    ignore_periods=params_by_dataset['dev']['ignore_periods']
                )
            
            # Log metrics
            logger.log_metrics(train_metrics, epoch, prefix='train')
            logger.log_metrics(dev_metrics, epoch, prefix='dev')
            logger.log_model_weights(model, epoch)
            
            # Log action distribution if available
            if 'actions' in train_metrics:
                logger.log_action_distribution(train_metrics['actions'], epoch)
            
            # Flush all metrics to wandb at once
            logger.flush_metrics()
            
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
                print(f'Epoch {epoch}: Train Loss = {train_metrics["loss/total"]:.4f}, '
                      f'Dev Loss = {dev_metrics["loss/total"]:.4f}')
            
            # Store the final metrics
            final_metrics.update(train_metrics)
            final_metrics.update({f"dev/{k}": v for k, v in dev_metrics.items()})
            
            # Early stopping if needed
            if dev_metrics['loss/reported'] < self.best_performance_data['dev_loss']:
                self.best_performance_data['dev_loss'] = dev_metrics['loss/reported']
                self.best_performance_data['epoch'] = epoch
        
        logger.close()
        return final_metrics

    def test(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, discrete_allocation=False):

        if model.policy.trainable and self.best_performance_data['model_params_to_save'] is not None:
            # Load the parameter weights that gave the best performance on the specified dataset
            model.policy.load_state_dict(self.best_performance_data['model_params_to_save'])

        average_test_loss, average_test_loss_to_report = self.do_one_epoch(
                optimizer, 
                data_loaders['test'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['test']['periods'], 
                problem_params, 
                observation_params, 
                train=True, 
                ignore_periods=params_by_dataset['test']['ignore_periods'],
                discrete_allocation=discrete_allocation
                )
        
        return {
            'loss/total': average_test_loss,
            'loss/reported': average_test_loss_to_report
        }

    def do_one_epoch(self, optimizer_wrapper, data_loader, loss_function, simulator, model, periods, problem_params, observation_params, train=True, ignore_periods=0, discrete_allocation=False):
        """
        Do one epoch of training or testing
        """
        
        epoch_loss = 0
        epoch_loss_to_report = 0  # Loss ignoring the first 'ignore_periods' periods
        total_samples = len(data_loader.dataset)
        periods_tracking_loss = periods - ignore_periods  # Number of periods for which we report the loss
        
        optimizer_metrics_sum = None
        num_batches = 0

        for i, data_batch in enumerate(data_loader):  # Loop through batches of data
            data_batch = self.move_batch_to_device(data_batch)
            
            # Forward pass and simulation
            total_reward, reward_to_report, trajectory_data = self.simulate_batch(
                loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods, discrete_allocation, collect_trajectories=True
                )
            
            # Always accumulate simulator metrics
            epoch_loss += total_reward.item()  # Rewards from period 0
            epoch_loss_to_report += reward_to_report.item()  # Rewards from period ignore_periods onwards
            
            # If training, get optimizer metrics but don't use them for loss tracking
            if train and model.trainable:
                batch_metrics = optimizer_wrapper.optimize(trajectory_data)
                
                # Accumulate optimizer metrics separately
                if optimizer_metrics_sum is None:
                    optimizer_metrics_sum = {k: v for k, v in batch_metrics.items()}
                else:
                    for k, v in batch_metrics.items():
                        optimizer_metrics_sum[k] += v
                num_batches += 1

        # Calculate average metrics using simulator results
        metrics = {
            'loss/total': epoch_loss/(total_samples*periods*problem_params['n_stores']),
            'loss/reported': epoch_loss_to_report/(total_samples*periods_tracking_loss*problem_params['n_stores'])
        }
        
        # Add optimizer metrics if available
        if optimizer_metrics_sum is not None:
            for k, v in optimizer_metrics_sum.items():
                if k not in ['loss/total', 'loss/reported']:  # Don't overwrite simulator metrics
                    metrics[k] = v / num_batches

        return metrics
    
    def simulate_batch(self, loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods=0, discrete_allocation=False, collect_trajectories=False):
        """
        Simulate for an entire batch of data, across the specified number of periods.
        Collects data for both HDPO (pathwise gradients) and optionally PPO (trajectory data).
        
        Parameters:
        -----------
        ...
        collect_trajectories: bool
            If True, collect and return trajectory data needed for PPO.
            If False, return None for trajectory_data to save memory.
        """
        # Initialize rewards
        batch_reward = 0
        reward_to_report = 0

        # Get observation keys from value network config if it exists
        observation_keys = None
        if hasattr(model, 'value_net') and model.value_net is not None:
            observation_keys = model.value_net.observation_keys

        # Initialize trajectory storage only if needed
        trajectory_data = None
        if collect_trajectories:
            trajectory_data = {
                'observations': [],
                'rewards': [],
                'actions': [],
                'logits': [],
                'values': [],
                'terminated': []
            }

        observation, _ = simulator.reset(periods, problem_params, data_batch, observation_params)
        # # set a fixed seed for debugging
        # torch.manual_seed(0)
        # np.random.seed(0)
        
        for t in range(periods):
            # Store observation if collecting trajectories
            vectorized_obs = self.vectorize_observation(observation, observation_keys)
            # if vectorized_obs is not None:
            #     trajectory_data['observations'].append(vectorized_obs.detach().clone())

            # Add internal data to observation
            observation_and_internal_data = {k: v for k, v in observation.items()}
            observation_and_internal_data['internal_data'] = simulator._internal_data

            # Sample action and get policy outputs
            model_output = model(observation_and_internal_data)
            action_dict = model_output.get('action_dict')
            value = model_output.get('value', None)

            if discrete_allocation:
                action_dict = {key: val.round() for key, val in action_dict.items()}            

            # Execute environment step
            next_observation, reward, terminated, _, _ = simulator.step(observation, action_dict)
            total_reward = loss_function(None, action_dict, reward)

            # Store trajectory data with proper detaching and cloning
            if collect_trajectories:
                if vectorized_obs is not None:
                    trajectory_data['observations'].append(vectorized_obs.detach().clone())
                trajectory_data['actions'].append(action_dict['discrete_actions'].detach().clone())
                trajectory_data['logits'].append(action_dict['action_logits'].detach().clone())
                if value is not None:
                    trajectory_data['values'].append(value.detach().clone())
                trajectory_data['rewards'].append(reward.detach().clone())
                trajectory_data['terminated'].append(torch.tensor(terminated).detach().clone())

            #     # print [-1][0] of every list in trajectory_data
            #     for key, value in trajectory_data.items():
            #         if key != 'terminated':
            #             print(f'{key}: {value[-1][0]}')
            # print()
            # Update running rewards
            batch_reward += total_reward
            if t >= ignore_periods:
                reward_to_report += total_reward

            # Update observation
            observation = next_observation

            if terminated:
                break

        # Convert trajectory lists to tensors with additional debugging
        if collect_trajectories:
            trajectory_data = {
                k: torch.stack(v) if v[0] is not None else None 
                for k, v in trajectory_data.items()
            }

        # print the shape of the trajectory data
        # print(f'trajectory_data["observations"].shape: {trajectory_data["observations"].shape}')
        # print(f'trajectory_data["actions"].shape: {trajectory_data["actions"].shape}')
        # print(f'trajectory_data["logits"].shape: {trajectory_data["logits"].shape}')
        # print(f'trajectory_data["values"].shape: {trajectory_data["values"].shape}')
        # print(f'trajectory_data["rewards"].shape: {trajectory_data["rewards"].shape}')
        # print(f'trajectory_data["terminated"].shape: {trajectory_data["terminated"].shape}')
        
        trajectory_data['next_observation'] = observation

        return batch_reward, reward_to_report, trajectory_data

    def save_model(self, epoch, model, optimizer_wrapper, trainer_params):
        path = self.create_many_folders_if_not_exist_and_return_path(
            base_dir=trainer_params['base_dir'], 
            intermediate_folder_strings=trainer_params['save_model_folders']
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_wrapper.optimizer.state_dict(),
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
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.policy.state_dict())
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
        return action_dict['logits'].reshape(action_dict['logits'].shape[0], -1).detach()

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

