from src.algorithms.common.optimizer_wrappers.base_wrapper import BaseOptimizerWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridWrapper(BaseOptimizerWrapper):
   def __init__(self, model, optimizer, gradient_clip=None, weight_decay=0.0, device='cpu'):
       super().__init__(model, optimizer, device)
       self.gradient_clip = gradient_clip
       
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
           'logits': trajectory_data['logits'].reshape(batch_size, -1),
           'advantages': advantages.reshape(batch_size),
           'returns': returns.reshape(batch_size),
           'values': trajectory_data['values'].reshape(batch_size)
       }
       tensors = {k: v.detach() for k, v in tensors.items()}
       
       clip_coef = 0.2
       num_optimizer_epochs = 1
       minibatch_size = T*B//64
       target_kl = 0.015
       norm_adv = True
       clip_vloss = True
       ent_coef = 0.01
       vf_coef = 0.5
       max_grad_norm = 0.5
       
       total_loss = 0
       b_inds = torch.arange(batch_size, device=self.device)
       counter = 0
       
       for epoch in range(num_optimizer_epochs):
           print(f"\nEpoch {epoch}")
           perm = torch.randperm(batch_size, device=self.device)
           b_inds = b_inds[perm]
           
           for start in range(0, batch_size, minibatch_size):
               end = start + minibatch_size
               mb_inds = b_inds[start:end]
               
               mb_obs = tensors['observations'][mb_inds]
               newlogits, newvalue, entropy = self.model.get_logits_value_and_entropy(mb_obs)
               newlogits = newlogits.squeeze(1)
               
               # Calculate log ratio directly from logits
               logratio = newlogits - tensors['logits'][mb_inds]
               if counter == 0:
                    if not torch.allclose(newlogits, tensors['logits'][mb_inds]):
                        print(f"New logits shape: {newlogits.shape}")
                        print(f"Old logits shape: {tensors['logits'][mb_inds].shape}")
                        diff_indices = (newlogits != tensors['logits'][mb_inds]).nonzero(as_tuple=True)[0]
                        print(f"Different logits at indices: {diff_indices}")
                        print(f"New logits: {newlogits[diff_indices]}")
                        print(f"Old logits: {tensors['logits'][mb_inds][diff_indices]}")
                    assert torch.allclose(newlogits, tensors['logits'][mb_inds])
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
               
               mb_advantages = tensors['advantages'][mb_inds]
               if norm_adv:
                   mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
               
               mb_advantages = mb_advantages.unsqueeze(-1).expand_as(ratio)
               pg_loss1 = -mb_advantages * ratio
               pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
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
               loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
               
               self.optimizer.zero_grad(set_to_none=True)
               loss.backward()
               if max_grad_norm is not None:
                   nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
               self.optimizer.step()
               
               total_loss += loss.item()
           
           if target_kl is not None and approx_kl > target_kl:
               break
       
       return total_loss / (num_optimizer_epochs * (batch_size // minibatch_size))

   def compute_advantages(self, trajectory_data):
       gamma = 0.99
       gae_lambda = 0.95
       
       T, B = trajectory_data['rewards'].shape
       
       with torch.no_grad():
           next_value = self.model.value_net(trajectory_data['next_observation'])
           values = trajectory_data['values'].reshape(T, B)
           rewards = trajectory_data['rewards']
           advantages = torch.zeros_like(rewards, device=self.device)
           
           lastgaelam = 0
           for t in reversed(range(T)):
               nextvalues = next_value.squeeze() if t == T - 1 else values[t + 1]
               delta = rewards[t] + gamma * nextvalues - values[t]
               advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
           
       returns = advantages + values
       return advantages, returns