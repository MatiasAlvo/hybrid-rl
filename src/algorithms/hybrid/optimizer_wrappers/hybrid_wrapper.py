from src.algorithms.common.optimizer_wrappers.base_wrapper import BaseOptimizerWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class HybridWrapper(BaseOptimizerWrapper):
   def __init__(self, model, optimizer, gradient_clip=None, weight_decay=0.0, device='cpu', ppo_params=None):
       super().__init__(model, optimizer, device)
       self.gradient_clip = gradient_clip
       self.ppo_params = ppo_params or {}  # Use empty dict if no params provided
       
   def optimize(self, trajectory_data):
       # Move to device once
       trajectory_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in trajectory_data.items()}
       
       
       # Get PPO parameters and prepare data
       clip_coef = self.ppo_params.get('clip_coef', 0.2)
       num_ppo_epochs = self.ppo_params.get('num_epochs', 1)
       target_kl = self.ppo_params.get('target_kl', None)
       reward_scaling = self.ppo_params.get('reward_scaling', False)
       
       if reward_scaling:
           rewards_std = trajectory_data['rewards'].std()
           if rewards_std > 0:
               trajectory_data['rewards'] = trajectory_data['rewards'] / (rewards_std + 1e-8)
       
       # First compute advantages and returns
       advantages, returns = self.compute_advantages(trajectory_data)
       advantages, returns = advantages.detach(), returns.detach()
       
       T, B = advantages.shape
       batch_size = T * B
       minibatch_size = batch_size // self.ppo_params.get('num_minibatches', 4)
       
       # Get buffer size
       buffer_periods = self.ppo_params.get('buffer_periods', 0)
       
       # Calculate effective trajectory length after buffer
       effective_T = T - 2 * buffer_periods if buffer_periods > 0 else T
       
       if effective_T <= 0:
           raise ValueError(f"Buffer size {buffer_periods} too large for trajectory length {T}")
       
       # Create slice for effective trajectory
       effective_slice = slice(buffer_periods, T - buffer_periods) if buffer_periods > 0 else slice(None)
       
       # Flatten all batches, applying buffer
       tensors = {
           'observations': trajectory_data['observations'][effective_slice].reshape(effective_T * B, -1),
           'actions': trajectory_data['actions'][effective_slice].reshape(effective_T * B),
           'logits': trajectory_data['logits'][effective_slice].reshape(effective_T * B),
           'advantages': advantages[effective_slice].reshape(effective_T * B),
           'returns': returns[effective_slice].reshape(effective_T * B),
           'values': trajectory_data['values'][effective_slice].reshape(effective_T * B)
       }
       tensors = {k: v.detach() for k, v in tensors.items()}
       
       # Update batch size for minibatch computation
       batch_size = effective_T * B
       minibatch_size = batch_size // self.ppo_params.get('num_minibatches', 4)
       
       # Get parameters from config
       norm_adv = self.ppo_params.get('normalize_advantages', True)
       clip_vloss = self.ppo_params.get('clip_value_loss', False)
       ent_coef = self.ppo_params.get('entropy_coef', 0.01)
       vf_coef = self.ppo_params.get('value_function_coef', 0.5)
       max_grad_norm = self.ppo_params.get('max_grad_norm', 0.5)
       
       # Initialize metrics
       total_loss = 0
       total_v_loss = 0
       total_pg_loss = 0
       total_entropy_loss = 0
       clipfracs = []
       approx_kls = []
       
       # Track first and last values
       first_clipfrac = None
       first_approx_kl = None
       last_clipfrac = None
       last_approx_kl = None
       
       # Track at which epoch and batch we stopped
       early_stop_epoch = num_ppo_epochs
       early_stop_batch = None
       
       # Pre-compute correlation metrics and value diagnostics
       with torch.no_grad():
           # Apply buffer to diagnostics
           effective_slice = slice(buffer_periods, T - buffer_periods) if buffer_periods > 0 else slice(None)
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
           early_stage = slice(0, T//3)
           mid_stage = slice(T//3, 2*T//3)
           late_stage = slice(2*T//3, T)
           
           def compute_stage_metrics(stage_slice):
               stage_values = trajectory_data['values'][stage_slice].reshape(-1)
               stage_returns = returns[stage_slice].reshape(-1)
               mae = (stage_values - stage_returns).abs().mean()
               mse = ((stage_values - stage_returns) ** 2).mean()
               return {'mae': mae.item(), 'mse': mse.item()}
           
           stage_metrics = {
               'early': compute_stage_metrics(early_stage),
               'mid': compute_stage_metrics(mid_stage),
               'late': compute_stage_metrics(late_stage)
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
       
       b_inds = torch.arange(batch_size, device=self.device)
       
       for epoch in range(num_ppo_epochs):
           perm = torch.randperm(batch_size, device=self.device)
           b_inds = b_inds[perm]
           
           for start in range(0, batch_size, minibatch_size):
               end = start + minibatch_size
               mb_inds = b_inds[start:end]
               
               mb_obs = tensors['observations'][mb_inds]
               mb_actions = tensors['actions'][mb_inds]
               
               newlogits, newvalue, entropy = self.model.get_logits_value_and_entropy(mb_obs, mb_actions)
               
               logratio = newlogits - tensors['logits'][mb_inds].unsqueeze(-1)
               logratio = torch.clamp(logratio, -20, 20)
               ratio = logratio.exp()
               
               with torch.no_grad():
                   approx_kl = ((ratio - 1) - logratio).mean()
                   approx_kls.append(approx_kl.item())
                   
                   if first_approx_kl is None:
                       first_approx_kl = approx_kl.item()
                   last_approx_kl = approx_kl.item()
               
               mb_advantages = tensors['advantages'][mb_inds]
               if norm_adv:
                   mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
               
               mb_advantages = mb_advantages.unsqueeze(-1).expand_as(ratio)
               pg_loss1 = mb_advantages * ratio
               pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
               pg_loss = torch.max(pg_loss1, pg_loss2).mean()
               
               if clip_vloss:
                   raise NotImplementedError("Clip value loss not implemented")
                   v_loss_unclipped = (newvalue - tensors['returns'][mb_inds]) ** 2
                   v_clipped = tensors['values'][mb_inds] + torch.clamp(
                       newvalue - tensors['values'][mb_inds],
                       -clip_coef,
                       clip_coef,
                   )
                   v_loss_clipped = (v_clipped - tensors['returns'][mb_inds]) ** 2
                   v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
               else:
                   v_loss = 0.5 * ((newvalue.squeeze() - tensors['returns'][mb_inds]) ** 2).mean()
               
               entropy_loss = entropy.mean()
            #    loss = v_loss * vf_coef
               loss = pg_loss + v_loss * vf_coef
               
               self.optimizer.zero_grad(set_to_none=True)
               loss.backward()
               if max_grad_norm is not None:
                   nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
               self.optimizer.step()
               
               total_loss += loss.item()
               total_v_loss += v_loss.item()
               total_pg_loss += pg_loss.item()
               total_entropy_loss += entropy_loss.item()
               
               clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
               clipfracs.append(clipfrac)
               
               if first_clipfrac is None:
                   first_clipfrac = clipfrac
               last_clipfrac = clipfrac
               
               if target_kl is not None and approx_kl > target_kl:
                   early_stop_epoch = epoch
                   early_stop_batch = start // minibatch_size
                   break
       
       # Compute final metrics
       num_updates = (early_stop_epoch * (batch_size // minibatch_size)) + 1
       
       # Calculate explained variance
       with torch.no_grad():
           y_pred = tensors['values'].cpu().numpy()
           y_true = tensors['returns'].cpu().numpy()
           var_y = np.var(y_true)
           explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
       
       metrics = {
           'loss/total': total_loss / num_updates,
           'loss/value': total_v_loss / num_updates,
           'loss/policy': total_pg_loss / num_updates,
           'loss/entropy': total_entropy_loss / num_updates,
           'policy/approx_kl': np.mean(approx_kls),
           'policy/approx_kl_first': first_approx_kl,
           'policy/approx_kl_last': last_approx_kl,
           'policy/clipfrac': np.mean(clipfracs),
           'policy/clipfrac_first': first_clipfrac,
           'policy/clipfrac_last': last_clipfrac,
           'policy/explained_var': explained_var,
           'optimization/total_epochs': early_stop_epoch,
           'optimization/early_stop_batch': early_stop_batch if early_stop_batch is not None else -1,
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
       
       return metrics

   def compute_advantages(self, trajectory_data):
       # Add debugging for advantage computation
    #    print("\n=== Advantage Computation Diagnostics ===")
    #    print(f"Initial shapes - rewards: {trajectory_data['rewards'].shape}, values: {trajectory_data['values'].shape}")
       
       # Get parameters from config
       gamma = self.ppo_params.get('gamma', 0.95)
       gae_lambda = self.ppo_params.get('gae_lambda', 0.99)
       use_gae = self.ppo_params.get('use_gae', True)
       
       T, B = trajectory_data['rewards'].shape

       with torch.no_grad():
           next_value = self.model.value_net(trajectory_data['next_observation'])
        #    print(f"Next value shape: {next_value.shape}, mean: {next_value.mean():.3f}, std: {next_value.std():.3f}")
           values = trajectory_data['values'].reshape(T, B)
           rewards = trajectory_data['rewards']
           
           if use_gae:
               # Existing GAE implementation
               advantages = torch.zeros_like(rewards, device=self.device)
               lastgaelam = 0
               for t in reversed(range(T)):
                   nextvalues = next_value.squeeze() if t == T - 1 else values[t + 1]
                   delta = rewards[t] + gamma * nextvalues - values[t]
                   advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
               returns = advantages + values
           else:
               # New alternative implementation
               returns = torch.zeros_like(rewards, device=self.device)
               for t in reversed(range(T)):
                #    next_return = 0 if t == T - 1 else returns[t + 1]
                   next_return = next_value.squeeze() if t == T - 1 else returns[t + 1]
                   returns[t] = rewards[t] + gamma * next_return
               advantages = returns - values

        #    print("\n=== Final Advantage Stats ===")
        #    print(f"Advantages - mean: {advantages.mean():.3f}, std: {advantages.std():.3f}")
        #    print(f"Returns - mean: {returns.mean():.3f}, std: {returns.std():.3f}")

       return advantages, returns

