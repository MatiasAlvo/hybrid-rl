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
       
       advantages, returns = self.compute_advantages(trajectory_data)
       advantages, returns = advantages.detach(), returns.detach()
       
       T, B = advantages.shape
       batch_size = T * B

       
       # Flatten all batches
       tensors = {
           'observations': trajectory_data['observations'].reshape(batch_size, -1),
           'actions': trajectory_data['actions'].reshape(batch_size),  # Actions are indices
           'logits': trajectory_data['logits'].reshape(batch_size),    # Logits for selected actions
           'advantages': advantages.reshape(batch_size),
           'returns': returns.reshape(batch_size),
           'values': trajectory_data['values'].reshape(batch_size)
       }
    #    print(f'tensors["observations"].shape: {tensors["observations"].shape}')

       tensors = {k: v.detach() for k, v in tensors.items()}
       
       # Use parameters from config instead of hardcoded values
       clip_coef = self.ppo_params.get('clip_coef', 0.2)
       num_ppo_epochs = self.ppo_params.get('num_epochs', 10)
       minibatch_size = T*B // self.ppo_params.get('num_minibatches', 4)
       target_kl = self.ppo_params.get('target_kl', 0.015)
       norm_adv = self.ppo_params.get('normalize_advantages', False)
       clip_vloss = self.ppo_params.get('clip_value_loss', False)
       ent_coef = self.ppo_params.get('entropy_coef', 0.01)
       vf_coef = self.ppo_params.get('value_function_coef', 0.5)
       max_grad_norm = self.ppo_params.get('max_grad_norm', 0.5)
       
       total_loss = 0
       total_v_loss = 0  # Track value loss separately
       total_pg_loss = 0
       total_entropy_loss = 0
       b_inds = torch.arange(batch_size, device=self.device)
       
       clipfracs = []
       explained_vars = []
       state_values_correlations = []
       value_return_correlations = []
       approx_kls = []
       early_stop_epochs = []
       
       # Track at which epoch we stopped
       early_stop_epoch = num_ppo_epochs  # Default to max epochs if no early stopping
       
       for epoch in range(num_ppo_epochs):
           counter = 0
           epoch_v_loss = 0
           num_minibatches = 0
           perm = torch.randperm(batch_size, device=self.device)
           b_inds = b_inds[perm]
           
           for start in range(0, batch_size, minibatch_size):
               end = start + minibatch_size
               mb_inds = b_inds[start:end]
               num_minibatches += 1
               
               mb_obs = tensors['observations'][mb_inds]
               mb_actions = tensors['actions'][mb_inds]
               
               
               # Get logits for specific actions, value, and entropy
               newlogits, newvalue, entropy = self.model.get_logits_value_and_entropy(mb_obs, mb_actions)
               
               # Calculate log ratio directly from logits
               logratio = newlogits - tensors['logits'][mb_inds].unsqueeze(-1)
               if counter == 0 and False:
                    print(f'newlogits: {newlogits[0]}')
                    print(f'tensors["logits"][mb_inds]: {tensors["logits"][mb_inds][0]}')
                    print(f'logratio: {logratio[0]}')
                    counter += 1
               
               # Debug log ratios before exp
               if torch.isnan(logratio).any() or torch.isinf(logratio).any():
                   print("\nWarning: Extreme values in log ratio before exp:")
                   print(f"Log ratio range: [{logratio.min().item():.3f}, {logratio.max().item():.3f}]")
                   print(f"Log ratio mean: {logratio.mean().item():.3f}")
                   print(f"New logits range: [{newlogits.min().item():.3f}, {newlogits.max().item():.3f}]")
                   print(f"Old logits range: [{tensors['logits'][mb_inds].min().item():.3f}, {tensors['logits'][mb_inds].max().item():.3f}]")
               
               # Clip log ratios to prevent extreme values
               logratio = torch.clamp(logratio, -20, 20)  # exp(20) is already very large
               ratio = logratio.exp()
               
               with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())

               mb_advantages = tensors['advantages'][mb_inds]
               if norm_adv:
                   mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
               
               mb_advantages = mb_advantages.unsqueeze(-1).expand_as(ratio)
               pg_loss1 = mb_advantages * ratio
               pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
               pg_loss = torch.max(pg_loss1, pg_loss2).mean()
               
               if clip_vloss:
                   v_loss_unclipped = (newvalue - tensors['returns'][mb_inds]) ** 2
                   v_clipped = tensors['values'][mb_inds] + torch.clamp(
                       newvalue - tensors['values'][mb_inds],
                       -clip_coef,
                       clip_coef,
                   )
                   v_loss_clipped = (v_clipped - tensors['returns'][mb_inds]) ** 2
                   v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
               else:
                   v_loss = 0.5 * ((newvalue - tensors['returns'][mb_inds]) ** 2).mean()                   
               
               entropy_loss = entropy.mean()
            #    loss = v_loss * vf_coef
               loss = pg_loss + v_loss * vf_coef
            #    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
               
               self.optimizer.zero_grad(set_to_none=True)
               loss.backward()
               if max_grad_norm is not None:
                   nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
               self.optimizer.step()
               
               epoch_v_loss += v_loss.item()
               
               total_loss += loss.item()
               total_v_loss += v_loss.item()
               total_pg_loss += pg_loss.item()
               total_entropy_loss += entropy_loss.item()

           if target_kl is not None and approx_kl > target_kl:
               early_stop_epochs.append(epoch)  # Record when we stopped
               print(f"Early stopping at epoch {epoch} due to KL divergence of {approx_kl}.")
               break
       
       # Check if different states get different values
       state_values_correlation = torch.corrcoef(torch.stack([
           tensors['observations'].sum(dim=1),  # Simplistic state representation
           tensors['values'].squeeze()
       ]))[0,1]
       state_values_correlations.append(state_values_correlation.item())
       rewards = trajectory_data['rewards'][20:50]
       discounted_rewards = torch.zeros_like(rewards[0:20])
       for t in range(20):
           discounted_rewards[t] = rewards[t:t+10].sum(dim=0)
       values = trajectory_data['values'][20:40]
       value_return_correlation = torch.corrcoef(torch.stack([
               values.flatten(), 
               discounted_rewards.flatten()
           ]))[0,1].item()
       value_return_correlations.append(value_return_correlation)
       # Calculate clipfrac
       clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
       clipfracs.append(clipfrac)

       
       # Calculate explained variance
       with torch.no_grad():
           y_pred = newvalue.cpu().numpy()
           y_true = tensors['returns'][mb_inds].cpu().numpy()
           var_y = np.var(y_true)
           explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
           explained_vars.append(explained_var)
       
       # Calculate average metrics
       metrics = {
           'loss/total': total_loss / (num_ppo_epochs * (batch_size // minibatch_size)),
           'loss/value': total_v_loss / (num_ppo_epochs * (batch_size // minibatch_size)),
           'loss/policy': total_pg_loss / (num_ppo_epochs * (batch_size // minibatch_size)),
           'loss/entropy': total_entropy_loss / (num_ppo_epochs * (batch_size // minibatch_size)),
           'policy/approx_kl': np.mean(approx_kls),
           'policy/clipfrac': np.mean(clipfracs),
           'policy/explained_var': np.mean(explained_vars),
           'policy/state_value_correlation': np.mean(state_values_correlations),
           'policy/value_return_correlation': np.mean(value_return_correlations),
       }
       
       # Add early stopping info to metrics
       metrics['optimization/total_epochs'] = early_stop_epoch
       
       return metrics

   def compute_advantages(self, trajectory_data):
       # Get parameters from config
       gamma = self.ppo_params.get('gamma', 0.95)
       gae_lambda = self.ppo_params.get('gae_lambda', 0.99)
       use_gae = self.ppo_params.get('use_gae', True)
       
       T, B = trajectory_data['rewards'].shape

       # Standardize rewards more efficiently by computing mean and std only once
       rewards = trajectory_data['rewards']
       mean_rewards = rewards.mean()
       std_rewards = rewards.std() + 1e-8
       trajectory_data['rewards'] = (rewards - mean_rewards) / std_rewards

       with torch.no_grad():
           next_value = self.model.value_net(trajectory_data['next_observation'])
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
                   next_return = next_value.squeeze() if t == T - 1 else returns[t + 1]
                   returns[t] = rewards[t] + gamma * next_return
               advantages = returns - values

       return advantages, returns
