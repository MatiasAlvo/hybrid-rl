import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc
import wandb

import torch
import torch.nn.functional as F

class BaseOptimizerWrapper:
    """Base optimizer wrapper with all possible optimization methods"""
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def compute_policy_loss(self, log_probs, advantages):
        """Policy gradient loss"""
        return -(log_probs * advantages).mean()
    
    def compute_value_loss(self, values, returns):
        """Value function loss"""
        return F.mse_loss(values, returns)
    
    def compute_entropy_loss(self, distribution):
        """Entropy bonus loss"""
        return -distribution.entropy().mean()
    
    def clip_gradients(self, max_grad_norm):
        """Gradient clipping utility"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
    
    def optimize(self, trajectory_data, rewards):
        """Each algorithm implements its specific optimization logic"""
        raise NotImplementedError 

class HybridWrapper(BaseOptimizerWrapper):
    def __init__(self, model, optimizer, device='cpu', ppo_params=None):
        super().__init__(model, optimizer, device)
        self.ppo_params = ppo_params or {}  # Use empty dict if no params provided
        # self.epoch_count = 0
        
        # Get required losses from the model
        self.required_losses = model.required_losses

    def optimize(self, trajectory_data):
        # Move to device once
        trajectory_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in trajectory_data.items()}

        # detach the observations
        trajectory_data['observations'] = trajectory_data['observations']
        # flip the sign of the rewards
        trajectory_data['rewards'] = -trajectory_data['rewards']

        # Process and prepare trajectory data (mostly parameters that will be used in training)
        processed_data = self._prepare_trajectory_data(trajectory_data)
        
        # Compute advantages and returns only if needed
        advantages, returns = None, None
        if self.required_losses['policy_gradient'] or self.required_losses['value']:
            advantages, returns = self.compute_advantages(trajectory_data)
        
        # Prepare tensors for training
        tensors = self._prepare_training_tensors(trajectory_data, advantages, returns, processed_data)
        
        # Compute pre-training diagnostics only if value network is used
        diagnostics = {}
        if self.required_losses['value'] and 'values' in trajectory_data:
            diagnostics = self._compute_diagnostics(trajectory_data, returns, tensors, processed_data)
        
        # Train the model
        training_metrics = self._train_model(tensors, processed_data)
        
        # Combine all metrics
        metrics = {**diagnostics, **training_metrics}
        
        # Clean up
        self._cleanup()
        
        return metrics

    def _prepare_trajectory_data(self, trajectory_data):
        """Process trajectory data and extract parameters needed for training."""
        processed_data = {}
        
        # Extract PPO parameters
        # Extract PPO parameters - all required, no defaults
        processed_data['clip_coef'] = self.ppo_params['clip_coef']
        processed_data['num_ppo_epochs'] = self.ppo_params['num_epochs']
        processed_data['pathwise_coef'] = self.ppo_params['pathwise_coef']
        processed_data['norm_adv'] = self.ppo_params['normalize_advantages']
        processed_data['vf_coef'] = self.ppo_params['value_function_coef']
        processed_data['max_grad_norm'] = self.ppo_params['max_grad_norm']

        # PPO parameters with defaults
        processed_data['target_kl'] = self.ppo_params.get('target_kl', 0.015)
        processed_data['clip_vloss'] = self.ppo_params.get('clip_value_loss', False)
        processed_data['ent_coef'] = self.ppo_params['entropy_coef']
       
        # Create a copy of rewards for pathwise computation only if needed
        if self.required_losses['pathwise']:
            processed_data['pathwise_rewards'] = trajectory_data['rewards'].clone()
            if self.ppo_params['reward_scaling_pathwise']:
                rewards_std = processed_data['pathwise_rewards'].std().detach()
                if rewards_std > 0:
                    processed_data['pathwise_rewards'] = processed_data['pathwise_rewards'] / (rewards_std + 1e-8)
        
        # Get buffer size and effective trajectory length
        buffer_periods = self.ppo_params.get('buffer_periods', 0)
        T, B = trajectory_data['rewards'].shape
        effective_T = T - 2 * buffer_periods if buffer_periods > 0 else T
        
        if effective_T <= 0:
            raise ValueError(f"Buffer size {buffer_periods} too large for trajectory length {T}")
        
        processed_data['T'] = T
        processed_data['B'] = B
        processed_data['buffer_periods'] = buffer_periods
        processed_data['effective_T'] = effective_T
        processed_data['effective_slice'] = slice(buffer_periods, T - buffer_periods) if buffer_periods > 0 else slice(None)
        
        # Store required losses in processed data for use in training
        processed_data['required_losses'] = self.required_losses
        
        return processed_data

    def _prepare_training_tensors(self, trajectory_data, advantages, returns, processed_data):
        """Prepare tensors for training."""
        # Extract data from trajectory
        observations = trajectory_data['observations']
        
        effective_slice = processed_data['effective_slice']
        effective_T = processed_data['effective_T']
        B = processed_data['B']
        
        # Flatten all batches, applying buffer
        tensors = {
            'observations': trajectory_data['observations'][effective_slice].reshape(effective_T * B, -1),
        }
        
        # Always include log_probs for histogram creation if available and not None
        if 'log_probs' in trajectory_data and trajectory_data['log_probs'] is not None:
            tensors['log_probs'] = trajectory_data['log_probs'][effective_slice].reshape(effective_T * B, -1)
        
        # Only include tensors needed for the required losses
        if self.required_losses['policy_gradient']:
            tensors.update({
                'discrete_action_indices': trajectory_data['discrete_action_indices'][effective_slice].reshape(effective_T * B),
                'advantages': advantages[effective_slice].reshape(effective_T * B),
            })
            
            # Add continuous samples if available (for GaussianPPOAgent)
            if 'raw_continuous_samples' in trajectory_data:
                tensors['raw_continuous_samples'] = trajectory_data['raw_continuous_samples'][effective_slice].reshape(-1, *trajectory_data['raw_continuous_samples'].shape[2:])
                # tensors['raw_continuous_samples'] = trajectory_data['raw_continuous_samples'][effective_slice].reshape(-1, trajectory_data['raw_continuous_samples'].shape[2], trajectory_data['raw_continuous_samples'].shape[3])
        
        if self.required_losses['value'] and 'values' in trajectory_data:
            tensors.update({
                'returns': returns[effective_slice].reshape(effective_T * B),
                'values': trajectory_data['values'][effective_slice].reshape(effective_T * B)
            })
        
        # Only detach tensors that shouldn't track gradients for pathwise derivatives
        if 'advantages' in tensors:
            tensors['advantages'] = tensors['advantages'].detach()
        if 'returns' in tensors:
            tensors['returns'] = tensors['returns'].detach()
        
        return tensors

    def _compute_diagnostics(self, trajectory_data, returns, tensors, processed_data):
        """Compute pre-training diagnostics and correlation metrics."""
        # Skip diagnostics if value function is not used
        if not self.required_losses['value'] or 'values' not in trajectory_data:
            return {}
        
        with torch.no_grad():
            # effective_slice = tensors['observations'].shape[0] // tensors['returns'].shape[0] if 'returns' in tensors else slice(None)
            effective_slice = processed_data['effective_slice']
            flat_values = trajectory_data['values'][effective_slice].reshape(-1)
            flat_returns = returns[effective_slice].reshape(-1)
            
            # Returns Correlation
            returns_corr = torch.corrcoef(torch.stack([flat_values, flat_returns]))[0,1].item()
            
            # R² Score
            mean_returns = flat_returns.mean()
            ss_tot = ((flat_returns - mean_returns) ** 2).sum()
            ss_res = ((flat_returns - flat_values) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot)
            
            # Value Error Analysis by Episode Stage
            T = trajectory_data['values'].shape[0]
            early_stage = slice(0, T//3)
            mid_stage = slice(T//3, 2*T//3)
            late_stage = slice(2*T//3, T)
            
            stage_metrics = {
                'early': self._compute_stage_metrics(trajectory_data, returns, early_stage),
                'mid': self._compute_stage_metrics(trajectory_data, returns, mid_stage),
                'late': self._compute_stage_metrics(trajectory_data, returns, late_stage)
            }
            
            # Value prediction distribution analysis
            value_stats = {
                'mean': flat_values.mean().item(),
                'std': flat_values.std().item(),
                'min': flat_values.min().item(),
                'max': flat_values.max().item(),
            }
            returns_stats = {
                'mean': flat_returns.mean().item(),
                'std': flat_returns.std().item(),
                'min': flat_returns.min().item(),
                'max': flat_returns.max().item(),
            }
            
            diagnostics = {
                'value/returns_correlation': returns_corr,
                'value/r2_score': r2_score,
                'value/pred_mean': value_stats['mean'],
                'value/pred_std': value_stats['std'],
                'value/returns_mean': returns_stats['mean'],
                'value/returns_std': returns_stats['std'],
                'value/early_stage_mae': stage_metrics['early']['mae'],
                'value/mid_stage_mae': stage_metrics['mid']['mae'],
                'value/late_stage_mae': stage_metrics['late']['mae'],
                'value/early_stage_mse': stage_metrics['early']['mse'],
                'value/mid_stage_mse': stage_metrics['mid']['mse'],
                'value/late_stage_mse': stage_metrics['late']['mse'],
            }
            
            return diagnostics

    def _compute_stage_metrics(self, trajectory_data, returns, stage_slice):
        """Compute metrics for a specific stage of the episode."""
        with torch.no_grad():
            stage_values = trajectory_data['values'][stage_slice].reshape(-1)
            stage_returns = returns[stage_slice].reshape(-1)
            mae = (stage_values - stage_returns).abs().mean()
            mse = ((stage_values - stage_returns) ** 2).mean()
            return {'mae': mae.item(), 'mse': mse.item()}

    def _train_model(self, tensors, processed_data):
        """Train the model using PPO with pathwise gradients."""
        # Initialize training metrics
        training_metrics = {}
        
        # Training parameters
        batch_size = tensors['observations'].shape[0]
        # batch indices
        b_inds = torch.arange(batch_size, device=self.device)
        num_ppo_epochs = processed_data['num_ppo_epochs']
        
        # Update Gumbel-Softmax temperature if applicable
        if hasattr(self.model, 'update_temperature'):
            current_temp = self.model.update_temperature()
            training_metrics['gumbel/temperature'] = current_temp
        
        # Initialize loss and policy metrics
        total_loss, total_v_loss, total_pg_loss = 0, 0, 0
        total_entropy_loss, total_pathwise_loss = 0, 0
        total_raw_entropy = 0  # Track raw entropy (without coefficient)
        clipfracs, approx_kls = [], []
        first_clipfrac, first_approx_kl = None, None
        last_clipfrac, last_approx_kl = None, None
        
        # Track early stopping
        early_stop_epoch = num_ppo_epochs
        early_stop_batch = None
        
        # Gradient metrics for analysis
        gradient_metrics = {}
        
        # Training loop
        for epoch in range(num_ppo_epochs):
            # For first epoch, use full batch
            current_minibatch_size = batch_size if epoch == 0 else batch_size // self.ppo_params.get('num_minibatches', 4)
            
            # shuffle batch indices
            perm = torch.randperm(batch_size, device=self.device)
            b_inds = b_inds[perm]
            
            # Track approx_kl for the epoch
            approx_kl = None
            
            # minibatch loop
            minibatch_idx = 0
            for start in range(0, batch_size, current_minibatch_size):
                end = start + current_minibatch_size
                mb_inds = b_inds[start:end]
                
                # Get minibatch data
                mb_data = self._get_minibatch(tensors, mb_inds)
                
                # Compute losses ALREADY MULTIPLIED by the coefficients
                policy_loss, value_loss, entropy_loss, pathwise_loss, metrics, raw_entropy = self._compute_losses(
                    mb_data, 
                    processed_data, 
                    epoch
                )
                
                # Update metrics
                clipfrac = metrics['clipfrac'] # fraction of action logits that were clipped
                approx_kl = metrics['approx_kl'] # KL divergence between new and old policy
                
                # Store first metrics
                if first_clipfrac is None:
                    first_clipfrac = clipfrac
                    if first_clipfrac > 0.0001:
                        raise ValueError("First clipfrac is greater than 0.0")
                    first_approx_kl = approx_kl.item()
                    if first_approx_kl > 0.0001:
                        print(f"First approx_kl: {first_approx_kl}")
                        raise ValueError("First approx_kl is greater than 0.0")
                    
                # Update last metrics
                last_clipfrac = clipfrac
                last_approx_kl = approx_kl.item()
                
                # Check gradient information on first pass (note that losses are already multiplied by the coefficients)
                if epoch == 0 and start == 0:
                    gradient_metrics = self._analyze_gradients(
                        policy_loss, 
                        value_loss, 
                        entropy_loss, 
                        pathwise_loss, 
                        processed_data
                    )
                
                # Perform optimization step
                batch_grad_metrics = self._optimization_step(
                    policy_loss, 
                    value_loss, 
                    entropy_loss, 
                    pathwise_loss, 
                    processed_data
                )
                
                # Update gradient metrics
                if batch_grad_metrics:
                    for k, v in batch_grad_metrics.items():
                        if k not in gradient_metrics:
                            gradient_metrics[k] = []
                        gradient_metrics[k].append(v)
                
                # After first epoch's update, detach observations and pathwise rewards
                # this is because pathwise gradients can only be used once, or otherwise an error is raised
                if epoch == 0 and start + current_minibatch_size >= batch_size:
                    tensors['observations'] = tensors['observations'].detach()
                    if 'pathwise_rewards' in processed_data:
                        processed_data['pathwise_rewards'] = processed_data['pathwise_rewards'].detach()
                
                # Update totals
                # note that we are now considering entropy loss in the total loss!!
                total_loss += policy_loss.item() + value_loss.item() + pathwise_loss.item() - entropy_loss.item()
                total_v_loss += value_loss.item()
                total_pg_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_pathwise_loss += pathwise_loss.item()
                
                # Track raw entropy if entropy is being computed
                if self.required_losses['entropy'] and raw_entropy is not None:
                    total_raw_entropy += raw_entropy.item()
                
                clipfracs.append(clipfrac)
                approx_kls.append(approx_kl.item())
                
                # Increment minibatch counter
                minibatch_idx += 1
                
                # Check for early stopping criteria
                # if the approx KL divergence is greater than the target KL threshold, stop training
                if processed_data['target_kl'] is not None and approx_kl > processed_data['target_kl']:
                    early_stop_epoch = epoch
                    early_stop_batch = minibatch_idx - 1  # Index of the batch that triggered early stop (0-indexed)
                    break  # Break out of minibatch loop
            
            # Break out of epoch loop if we hit KL divergence limit
            if early_stop_epoch < num_ppo_epochs and early_stop_batch is not None:
                break
        
        if self.required_losses['value']:
            # Compute explained variance
            explained_var = self._compute_explained_variance(tensors)
        
        # Compute number of updates for averaging
        # Epoch 0 always has 1 update (full batch), subsequent epochs have num_minibatches updates
        num_minibatches = self.ppo_params.get('num_minibatches', 4)
        
        if early_stop_batch is not None:
            # Early stopping occurred - count all completed epochs plus partial minibatches
            # Epoch 0 has 1 update
            num_updates = 1
            # Add updates from epochs 1 to early_stop_epoch-1 (if any)
            if early_stop_epoch > 1:
                num_updates += (early_stop_epoch - 1) * num_minibatches
            # Add updates from the early_stop_epoch (minibatches 0 to early_stop_batch inclusive)
            if early_stop_epoch > 0:
                num_updates += early_stop_batch + 1
        else:
            # No early stopping - completed all epochs
            # Epoch 0 has 1 update, epochs 1 to num_ppo_epochs-1 have num_minibatches each
            num_updates = 1 + (num_ppo_epochs - 1) * num_minibatches
        
        # Compile metrics
        metrics = {
            'loss/total': total_loss / num_updates,
            'loss/value': total_v_loss / num_updates,
            'loss/policy': total_pg_loss / num_updates,
            'loss/pathwise': total_pathwise_loss / num_updates,
            'loss/entropy': total_entropy_loss / num_updates,
        }
        
        # Only include policy metrics if policy gradient is used
        if self.required_losses['policy_gradient']:
            metrics.update({
                'policy/approx_kl': np.mean(approx_kls),
                'policy/approx_kl_first': first_approx_kl,
                'policy/approx_kl_last': last_approx_kl,
                'policy/clipfrac': np.mean(clipfracs),
                'policy/clipfrac_first': first_clipfrac,
                'policy/clipfrac_last': last_clipfrac,
            })
        
        # Only include explained variance if value function is used
        if self.required_losses['value'] and 'returns' in tensors:
            metrics['policy/explained_var'] = explained_var
        
        metrics.update({
            'optimization/total_epochs': early_stop_epoch,
            'optimization/early_stop_batch': early_stop_batch if early_stop_batch is not None else -1,
        })
        
        # Process gradient metrics (average across batches)
        if gradient_metrics:
            for k, v in gradient_metrics.items():
                if isinstance(v, list):
                    metrics[k] = np.mean(v)
                else:
                    metrics[k] = v
        
        # Add temperature to metrics if it exists
        if hasattr(self.model, 'temperature'):
            metrics['gumbel/temperature'] = self.model.temperature
        
        # Add raw entropy metric if entropy is being computed
        if self.required_losses['entropy']:
            metrics['entropy/mean'] = total_raw_entropy / num_updates
        
        return metrics

    def _get_minibatch(self, tensors, mb_inds):
        """Extract minibatch data from tensors."""
        mb_data = {'observations': tensors['observations'][mb_inds]}
        
        # Only include tensors that are present
        if 'discrete_action_indices' in tensors:
            mb_data['discrete_action_indices'] = tensors['discrete_action_indices'][mb_inds]
        
        if 'log_probs' in tensors:
            mb_data['log_probs'] = tensors['log_probs'][mb_inds]
        
        if 'advantages' in tensors:
            mb_data['advantages'] = tensors['advantages'][mb_inds]
        
        if 'returns' in tensors:
            mb_data['returns'] = tensors['returns'][mb_inds]
        
        if 'values' in tensors:
            mb_data['values'] = tensors['values'][mb_inds]
        
        if 'raw_continuous_samples' in tensors:
            mb_data['raw_continuous_samples'] = tensors['raw_continuous_samples'][mb_inds]
        
        return mb_data

    def _compute_losses(self, mb_data, processed_data, epoch):
        """Compute all loss components for the current minibatch."""
        required_losses = processed_data['required_losses']
        
        # Initialize losses and metrics
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        pathwise_loss = torch.tensor(0.0, device=self.device)
        metrics = {'clipfrac': 0.0, 'approx_kl': torch.tensor(0.0, device=self.device)}
        
        # Get new log_probs, value, and entropy if needed
        continuous_samples = mb_data.get('raw_continuous_samples', None)
        new_log_probs, newvalue, entropy = None, None, None
        
        # Only call get_log_probs_value_and_entropy if we need any of its outputs
        if required_losses['policy_gradient'] or required_losses['value'] or required_losses['entropy']:
            new_log_probs, newvalue, entropy = self.model.get_log_probs_value_and_entropy(
                mb_data['observations'], 
                mb_data.get('discrete_action_indices', None),
                continuous_samples
            )
        
        # Compute policy gradient loss if needed
        if required_losses['policy_gradient'] and new_log_probs is not None:
            # Compute log ratio and ratio for PPO
            logratio = new_log_probs - mb_data['log_probs']
            if new_log_probs.shape[0] != mb_data['log_probs'].shape[0] or new_log_probs.shape[1] != mb_data['log_probs'].shape[1]:
                raise ValueError("Log ratio shape mismatch. shapes: ", new_log_probs.shape, mb_data['log_probs'].shape)
            # logratio = new_log_probs - mb_data['log_probs'].unsqueeze(-1) # change March 18
            logratio = torch.clamp(logratio, -20, 20)
            ratio = torch.exp(logratio)
            
            # Compute approx KL divergence
            with torch.no_grad():
                metrics['approx_kl'] = ((ratio - 1) - logratio).mean()
            
            # Handle advantage normalization
            mb_advantages = mb_data['advantages']
            if processed_data['norm_adv']:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Expand advantages to match ratio shape
            mb_advantages = mb_advantages.unsqueeze(-1).expand_as(ratio)
            
            # Compute policy loss (PPO style)
            clip_coef = processed_data['clip_coef']
            pg_loss1 = mb_advantages * ratio
            pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            # policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
            
            # Compute clipping fraction for diagnostics
            metrics['clipfrac'] = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
        
        # Compute value loss if needed
        if required_losses['value'] and newvalue is not None:
            if processed_data['clip_vloss']:
                v_loss_unclipped = (newvalue.squeeze() - mb_data['returns']) ** 2
                v_clipped = mb_data['values'] + torch.clamp(
                    newvalue.squeeze() - mb_data['values'],
                    -processed_data['clip_coef'],
                    processed_data['clip_coef'],
                )
                v_loss_clipped = (v_clipped - mb_data['returns']) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                res = (newvalue.squeeze() - mb_data['returns'].detach())
                res_squared = res ** 2
                value_loss = 0.5 * ((newvalue.squeeze() - mb_data['returns'].detach()) ** 2).mean()
            value_loss = processed_data['vf_coef'] * value_loss
        
        # Compute entropy loss if needed
        raw_entropy = None
        if required_losses['entropy'] and entropy is not None:
            raw_entropy = entropy.mean()
            entropy_loss = raw_entropy
            entropy_loss = processed_data['ent_coef'] * entropy_loss
        
        # Add pathwise loss component for continuous actions if needed
        if required_losses['pathwise'] and 'pathwise_rewards' in processed_data:
            pathwise_rewards_slice = processed_data['pathwise_rewards'][processed_data['effective_slice']]
            pathwise_loss = - pathwise_rewards_slice.mean()
            pathwise_loss = processed_data['pathwise_coef'] * pathwise_loss
        
        return policy_loss, value_loss, entropy_loss, pathwise_loss, metrics, raw_entropy

    def _analyze_gradients(self, policy_loss, value_loss, entropy_loss, pathwise_loss, processed_data):
        """Analyze gradients from different loss components."""
        gradient_metrics = {}
        required_losses = processed_data['required_losses']
        
        # Only analyze pathwise gradients if pathwise loss is used
        if required_losses['pathwise'] and pathwise_loss.requires_grad:
            pathwise_loss.backward(retain_graph=True)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if any(key in name for key in ['continuous', 'backbone', 'discrete']):
                        gradient_metrics[f'grad_analysis/pathwise/{name}'] = param.grad.abs().mean().item()
        
        # Zero gradients before checking other losses
        self.optimizer.zero_grad()
        
        # Only analyze policy gradients if policy gradient loss is used
        if required_losses['policy_gradient'] and policy_loss.requires_grad:
            policy_loss.backward(retain_graph=True)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if any(key in name for key in ['continuous', 'discrete', 'backbone']):
                        gradient_metrics[f'grad_analysis/policy/{name}'] = param.grad.abs().mean().item()
        
        # Zero gradients again
        self.optimizer.zero_grad()
        
        # Only analyze value gradients if value loss is used
        if required_losses['value'] and value_loss.requires_grad:
            value_loss.backward(retain_graph=True)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if any(key in name for key in ['value', 'backbone']):
                        gradient_metrics[f'grad_analysis/value/{name}'] = param.grad.abs().mean().item()
        
        # Zero gradients again
        self.optimizer.zero_grad()
        
        return gradient_metrics

    def _optimization_step(self, policy_loss, value_loss, entropy_loss, pathwise_loss, processed_data):
        """Perform optimization step with only the required losses combined."""
        self.optimizer.zero_grad()
        
        # Combine only the required losses. Note that losses are already multiplied by the coefficients.
        required_losses = processed_data['required_losses']
        loss = torch.tensor(0.0, device=self.device)
        
        if required_losses['policy_gradient']:
            loss = loss + policy_loss
        
        if required_losses['value']:
            loss = loss + value_loss
        
        if required_losses['pathwise']:
            loss = loss + pathwise_loss
        
        if required_losses['entropy']:
            loss = loss - entropy_loss  # Negative because we want to maximize entropy
        
        # Skip backward if no loss components are used
        grad_metrics = {}
        if loss.requires_grad:
            loss.backward()
            
            # Apply gradient clipping if specified
            if processed_data['max_grad_norm'] is not None:
                grad_metrics = self._clip_gradients_by_component(processed_data['max_grad_norm'])
            
            # Perform optimizer step
            self.optimizer.step()
            
            # Optional: Verify parameter updates (uncomment for debugging)
            # if self._step_count % 50 == 0:  # Every 50 steps
            #     print(f"\n=== PARAMETER UPDATE VERIFICATION (Step {self._step_count}) ===")
            #     for i, param_group in enumerate(self.optimizer.param_groups):
            #         group_name = param_group.get('name', f'group_{i}')
            #         lr = param_group['lr']
            #         print(f"Group '{group_name}' (LR={lr:.6f}):")
            #         for j, param in enumerate(param_group['params'][:2]):  # Show first 2 params
            #             if param.grad is not None:
            #                 grad_norm = param.grad.norm().item()
            #                 param_norm = param.norm().item()
            #                 print(f"  Param {j}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}")
            #     print("=" * 60)
        
        return grad_metrics
    def _clip_gradients_by_component_debug(self, max_grad_norm):
        """Clip gradients with a fixed maximum norm."""
        print()
        print(f"=== CLIP FUNCTION CALLED ===")
        print(f"max_grad_norm type: {type(max_grad_norm)}")
        print(f"max_grad_norm value: {max_grad_norm}")
        # Use a FIXED clipping threshold
        FIXED_MAX_NORM = 1.0  # or 0.5, or whatever you want

        # Add this right in your clipping function:
        print("Testing simple clipping:")
        all_params = [p for p in self.model.parameters() if p.grad is not None]
        if all_params:
            # Test 1: Get current norm
            current_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
            print(f"Current total norm: {current_norm}")
            
            # Test 2: Try to clip to 1.0
            clipped_norm = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            print(f"After clipping to 1.0: {clipped_norm}")
            
            # Test 3: Check if gradients actually changed
            current_norm_again = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
            print(f"Norm after checking again: {current_norm_again}")
        # Add this to see gradient info:
        for i, p in enumerate(all_params[:3]):  # Check first 3 parameters
            if p.grad is not None:
                print(f"Param {i}: grad shape {p.grad.shape}, grad norm {torch.norm(p.grad)}")
                print(f"Param {i}: grad has NaN: {torch.isnan(p.grad).any()}")
                print(f"Param {i}: grad has Inf: {torch.isinf(p.grad).any()}")
        
        # Get all parameters with gradients
        params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
        print(f"Params with grad: {len(params_with_grad)}")
        print(f"Params with grad: {params_with_grad}")
        
        if not params_with_grad:
            return {}
        
        # Calculate gradient norm before clipping
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in params_with_grad]))
        
        # Apply clipping with FIXED threshold
        clipped_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, FIXED_MAX_NORM)
        
        print(f"FIXED_MAX_NORM: {FIXED_MAX_NORM}")
        print(f"grad_norm_before: {total_norm.item()}")
        print(f"grad_norm_after: {clipped_norm.item()}")
        
        return {
            'grad_norm/total/before': total_norm.item(),
            'grad_norm/total/after': clipped_norm.item(),
            'grad_norm/total/clip_ratio': clipped_norm.item() / (total_norm.item() + 1e-8)
        }
    
    def _clip_gradients_by_component(self, max_grad_norm):
        """Clip gradients separately for each network component."""
        # Group parameters by network component
        param_groups = {
            'value': [],
            'backbone': [],
            'continuous': [],
            'discrete': [],
            'other': []
        }
        
        # Categorize parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'value' in name:
                    param_groups['value'].append(param)
                elif 'backbone' in name:
                    param_groups['backbone'].append(param)
                elif 'continuous' in name:
                    param_groups['continuous'].append(param)
                elif 'discrete' in name:
                    param_groups['discrete'].append(param)
                else:
                    param_groups['other'].append(param)
        # Track gradient norms for logging
        grad_metrics = {}
        
        # Clip gradients separately for each group
        for group_name, params in param_groups.items():
            if params:  # Only clip if group has parameters
                # Get parameters with non-None gradients
                params_with_grad = [p for p in params if p.grad is not None]
                
                # Skip if no parameters have gradients
                if not params_with_grad:
                    continue

                # Apply clipping and get the norm before clipping
                grad_norm_before = torch.nn.utils.clip_grad_norm_(params_with_grad, max_grad_norm)

                # Calculate norm after clipping
                grad_norms_after = [torch.norm(p.grad.detach()) for p in params_with_grad]
                grad_norm_after = torch.norm(torch.stack(grad_norms_after))


                
                # Store metrics
                grad_metrics[f'grad_norm/{group_name}/before'] = grad_norm_before.item()
                grad_metrics[f'grad_norm/{group_name}/after'] = grad_norm_after.item()
                grad_metrics[f'grad_norm/{group_name}/clip_ratio'] = grad_norm_after.item() / (grad_norm_before.item() + 1e-8)
        
        return grad_metrics

    def _compute_explained_variance(self, tensors):
        """Compute explained variance metric."""
        with torch.no_grad():
            y_pred = tensors['values'].cpu().numpy()
            y_true = tensors['returns'].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            return explained_var

    def _cleanup(self):
        """Clean up resources after training."""
        # Clear the autograd graph
        for param in self.model.parameters():
            param.grad = None  # More efficient than zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    def compute_advantages(self, trajectory_data):
        """Compute advantages and returns from trajectory data."""
        # Skip if not needed
        if not self.required_losses['policy_gradient'] and not self.required_losses['value']:
            return None, None
            
        rewards = trajectory_data['rewards'].clone().detach()
        if self.ppo_params['reward_scaling']:
            rewards_std = rewards.std()
            if rewards_std > 0:
                rewards = rewards / (rewards_std + 1e-8)
        
        # Get parameters from config
        gamma = self.ppo_params.get('gamma', 0.95)
        gae_lambda = self.ppo_params.get('gae_lambda', 0.99)
        use_gae = self.ppo_params.get('use_gae', True)
        
        T, B = trajectory_data['rewards'].shape
        
        with torch.no_grad():
            next_value = self.model.value_net(trajectory_data['next_observation'])
            values = trajectory_data['values'].reshape(T, B)
            
            if use_gae:
                # GAE implementation
                advantages = torch.zeros_like(rewards, device=self.device)
                lastgaelam = 0
                for t in reversed(range(T)):
                    nextvalues = next_value.squeeze() if t == T - 1 else values[t + 1]
                    delta = rewards[t] + gamma * nextvalues - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
                returns = advantages + values
            else:
                # Alternative implementation
                returns = torch.zeros_like(rewards, device=self.device)
                for t in reversed(range(T)):
                    next_return = next_value.squeeze() if t == T - 1 else returns[t + 1]
                    returns[t] = rewards[t] + gamma * next_return
                advantages = returns - values
            
            return advantages, returns

    def _create_discrete_probabilities_histogram(self, log_probs):
        """Create histogram data of discrete probabilities for the last action."""
        try:
            # Print shape information for debugging
            print(f"log_probs shape: {log_probs.shape}")
            
            # Apply softmax to get probabilities
            probs = F.softmax(log_probs, dim=-1)
            print(f"Probs shape after softmax: {probs.shape}")
            
            # Get the probability of the last action
            # Shape of log_probs is typically [batch_size, num_actions]
            try:
                # Try to access the last dimension
                last_action_probs = probs[:, -1].flatten().detach().cpu().numpy()
            except IndexError:
                # If that fails, try a different approach based on the actual shape
                if len(probs.shape) == 2:
                    # If shape is [batch, actions]
                    last_action_probs = probs[:, -1].detach().cpu().numpy()
                else:
                    # If shape is [batch, stores, actions]
                    last_action_probs = probs[:, 0, -1].detach().cpu().numpy()
            
            print(f"Last action probs shape: {last_action_probs.shape}")
            print(f"Last action probs sample: {last_action_probs[:5]}")
            
            # Create histogram data for wandb
            histogram = wandb.Histogram(last_action_probs)
            print(f"Successfully created wandb histogram")
            
            return {
                'policy/discrete_probs_histogram': histogram
            }
            
        except Exception as e:
            # Catch any errors to prevent training from failing
            print(f"Warning: Failed to create discrete probabilities histogram: {e}")
            import traceback
            traceback.print_exc()
            return {}