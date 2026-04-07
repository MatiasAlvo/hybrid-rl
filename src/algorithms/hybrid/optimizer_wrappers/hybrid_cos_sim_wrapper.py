import copy
import torch

from src.algorithms.hybrid.optimizer_wrappers.hybrid_wrapper import HybridWrapper


class HybridCosSimWrapper(HybridWrapper):
    """
    Optimizer wrapper that performs shadow cosine-similarity analysis for continuous
    parameter gradients. During shadow epochs (skip_update + cosine_grad_analysis),
    it computes per-batch gradients for:
      - PPO (score function only)
      - Hybrid with cross-term (policy + pathwise)
      - Hybrid without cross-term (discrete-only policy + pathwise)
    and compares them against true gradients aggregated across batches.
    """
    def __init__(
        self,
        model,
        optimizer,
        problem_params,
        device='cpu',
        ppo_params=None,
        base_lr=None,
        lr_multipliers=None
    ):
        super().__init__(
            model,
            optimizer,
            problem_params,
            device=device,
            ppo_params=ppo_params,
            base_lr=base_lr,
            lr_multipliers=lr_multipliers
        )
        self._shadow_grad_buffers = None
        self._shadow_norm_buffers = None
        self._last_shadow_metrics = {}
        self._continuous_params = [
            (name, param) for name, param in self.model.named_parameters()
            if 'continuous' in name
        ]
    
    def on_epoch_start(self):
        super().on_epoch_start()
        if self.cosine_grad_analysis and self.skip_update:
            self._shadow_grad_buffers = {
                'continuous_ppo': [],
                'continuous_reinforce': [],
                'pathwise_plus_cross_ppo': [],
                'pathwise_plus_cross_reinforce': []
            }
            self._shadow_norm_buffers = {
                'continuous_ppo': [],
                'continuous_reinforce': [],
                'pathwise_plus_cross_ppo': [],
                'pathwise_plus_cross_reinforce': []
            }
            self._shadow_alignment_batch_size = None
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if (self.cosine_grad_analysis and self.skip_update
                and self._shadow_grad_buffers is not None):
            self._last_shadow_metrics = self._compute_shadow_metrics()
            self._shadow_grad_buffers = None
            self._shadow_norm_buffers = None
    
    def get_epoch_metrics(self):
        metrics = super().get_epoch_metrics()
        if self._last_shadow_metrics:
            metrics.update(self._last_shadow_metrics)
            self._last_shadow_metrics = {}
        return metrics
    
    def optimize(self, trajectory_data):
        if not (self.cosine_grad_analysis and self.skip_update):
            return super().optimize(trajectory_data)
        self._collect_shadow_gradients(trajectory_data)
        return {}
    
    def _collect_shadow_gradients(self, trajectory_data):
        # Move to device once
        trajectory_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in trajectory_data.items()}
        
        # Match training reward convention
        trajectory_data['rewards'] = -trajectory_data['rewards']

        saved_required_losses = copy.deepcopy(self.required_losses)
        try:
            # Force full loss context for shadow grad analysis
            self.required_losses = {
                'policy_gradient': True,
                'value': True,
                'pathwise': True,
                'entropy': False
            }

            # Prepare data
            processed_data = self._prepare_trajectory_data(trajectory_data)
            advantages, returns, cum_rewards = self.compute_advantages(trajectory_data)
        finally:
            self.required_losses = saved_required_losses
        
        # Use training tensors (full batch)
        tensors = self._prepare_training_tensors(
            trajectory_data, advantages, returns, cum_rewards, processed_data
        )
        
        batch_size = tensors['observations'].shape[0]
        mb_inds = torch.arange(batch_size, device=self.device)
        mb_data = self._get_minibatch(tensors, mb_inds)

        # Prepare discrete-only log_probs (continuous log-prob removed)
        discrete_only_log_probs = self._compute_old_discrete_log_probs(
            trajectory_data, processed_data
        )

        if self._cosine_alignment_batch_size is None:
            self._cosine_alignment_batch_size = processed_data.get('B')
        if self._shadow_alignment_batch_size is None:
            self._shadow_alignment_batch_size = processed_data.get('B')

        # Populate base HybridWrapper cosine metrics with detached continuous log-probs
        self._analyze_cosine_gradients(
            mb_data,
            processed_data,
            discrete_only_log_probs
        )
        
        # Build per-estimator minibatch views
        mb_data_combined = dict(mb_data)
        mb_data_no_cont_logprob = dict(mb_data)
        if discrete_only_log_probs is not None:
            mb_data_no_cont_logprob['log_probs'] = discrete_only_log_probs
        mb_data_no_cont_logprob['raw_continuous_samples'] = None
        
        # Compute gradient vectors for each estimator
        self._collect_grad_vector(
            self._policy_loss_only(mb_data_combined, processed_data),
            'continuous_ppo',
            batch_size=processed_data.get('B')
        )
        self._collect_grad_vector(
            self._policy_plus_pathwise(mb_data_no_cont_logprob, processed_data),
            'pathwise_plus_cross_ppo',
            batch_size=processed_data.get('B')
        )
        self._collect_grad_vector(
            self._reinforce_plus_pathwise(mb_data_no_cont_logprob, processed_data),
            'pathwise_plus_cross_reinforce',
            batch_size=processed_data.get('B')
        )
        self._collect_grad_vector(
            self._reinforce_loss_only(mb_data_combined, processed_data),
            'continuous_reinforce',
            batch_size=processed_data.get('B')
        )
    
    def _compute_policy_loss(self, mb_data, processed_data, detach_obs=False, detach_continuous_log_probs=False):
        observations = mb_data['observations'].detach() if detach_obs else mb_data['observations']
        discrete_action_indices = mb_data.get('discrete_action_indices')
        continuous_samples = mb_data.get('raw_continuous_samples')
        new_log_probs_full, _, _ = self.model.get_log_probs_value_and_entropy(
            observations,
            discrete_action_indices,
            continuous_samples
        )
        if new_log_probs_full is None:
            return torch.tensor(0.0, device=self.device)
        
        new_log_probs = new_log_probs_full
        if detach_continuous_log_probs:
            new_log_probs_discrete, _, _ = self.model.get_log_probs_value_and_entropy(
                observations,
                discrete_action_indices,
                None
            )
            if new_log_probs_discrete is not None:
                new_log_probs = new_log_probs_discrete
        
        logratio = new_log_probs - mb_data['log_probs']
        logratio = torch.clamp(logratio, -20, 20)
        ratio = torch.exp(logratio)
        
        mb_advantages = mb_data['advantages']
        if processed_data['norm_adv']:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        mb_advantages = mb_advantages.unsqueeze(-1).expand_as(ratio)
        
        clip_coef = processed_data['clip_coef']
        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
        return policy_loss

    def _policy_loss_only(self, mb_data, processed_data):
        return self._compute_policy_loss(
            mb_data,
            processed_data,
            detach_obs=True,
            detach_continuous_log_probs=False
        )

    def _reinforce_loss_only(self, mb_data, processed_data):
        return self._compute_reinforce_loss(
            mb_data,
            processed_data,
            detach_obs=True,
            detach_continuous_log_probs=False
        )
    
    def _policy_plus_pathwise(self, mb_data, processed_data):
        policy_loss = self._compute_policy_loss(
            mb_data,
            processed_data,
            detach_obs=False,
            detach_continuous_log_probs=True
        )
        required = {
            'policy_gradient': False,
            'value': False,
            'pathwise': True,
            'entropy': False
        }
        processed = dict(processed_data)
        processed['required_losses'] = required
        processed['use_reinforce_loss'] = False
        _, _, _, pathwise_loss, _, _, _ = self._compute_losses(mb_data, processed, epoch=0)
        return policy_loss + pathwise_loss

    def _reinforce_plus_pathwise(self, mb_data, processed_data):
        reinforce_loss = self._compute_reinforce_loss(
            mb_data,
            processed_data,
            detach_obs=False,
            detach_continuous_log_probs=True
        )
        required = {
            'policy_gradient': False,
            'value': False,
            'pathwise': True,
            'entropy': False
        }
        processed = dict(processed_data)
        processed['required_losses'] = required
        processed['use_reinforce_loss'] = False
        _, _, _, pathwise_loss, _, _, _ = self._compute_losses(mb_data, processed, epoch=0)
        return reinforce_loss + pathwise_loss

    def _compute_reinforce_loss(self, mb_data, processed_data, detach_obs=False, detach_continuous_log_probs=False):
        if 'cum_rewards' not in mb_data or mb_data['cum_rewards'] is None:
            return torch.tensor(0.0, device=self.device)
        observations = mb_data['observations'].detach() if detach_obs else mb_data['observations']
        discrete_action_indices = mb_data.get('discrete_action_indices')
        continuous_samples = mb_data.get('raw_continuous_samples')
        new_log_probs_full, _, _ = self.model.get_log_probs_value_and_entropy(
            observations,
            discrete_action_indices,
            continuous_samples
        )
        if new_log_probs_full is None:
            return torch.tensor(0.0, device=self.device)
        new_log_probs = new_log_probs_full
        if detach_continuous_log_probs:
            new_log_probs_discrete, _, _ = self.model.get_log_probs_value_and_entropy(
                observations,
                discrete_action_indices,
                None
            )
            if new_log_probs_discrete is not None:
                new_log_probs = new_log_probs_discrete
        
        logratio = new_log_probs - mb_data['log_probs']
        logratio = torch.clamp(logratio, -20, 20)
        ratio = torch.exp(logratio)
        
        mb_returns = mb_data['cum_rewards']
        if processed_data['norm_adv']:
            mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
        mb_returns = mb_returns.unsqueeze(-1).expand_as(ratio)
        
        clip_coef = processed_data['clip_coef']
        reinforce_loss = -torch.min(
            mb_returns * ratio,
            mb_returns * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        ).mean()
        return reinforce_loss

    def _analyze_cosine_gradients(self, mb_data, processed_data, discrete_only_log_probs):
        required_losses = processed_data['required_losses']
        policy_loss = torch.tensor(0.0, device=self.device)
        reinforce_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        pathwise_loss = torch.tensor(0.0, device=self.device)
        mb_data_discrete_logprob = dict(mb_data)
        if discrete_only_log_probs is not None:
            mb_data_discrete_logprob['log_probs'] = discrete_only_log_probs
        
        if required_losses['policy_gradient']:
            policy_loss = self._compute_policy_loss(
                mb_data_discrete_logprob,
                processed_data,
                detach_obs=False,
                detach_continuous_log_probs=True
            )
        
        if processed_data.get('use_reinforce_loss', False):
            reinforce_loss = self._compute_reinforce_loss(
                mb_data_discrete_logprob,
                processed_data,
                detach_obs=False,
                detach_continuous_log_probs=True
            )
        
        if required_losses['value']:
            processed = dict(processed_data)
            processed['required_losses'] = {
                'policy_gradient': False,
                'value': True,
                'pathwise': False,
                'entropy': False
            }
            _, value_loss, _, _, _, _, _ = self._compute_losses(mb_data, processed, epoch=0)
        
        if required_losses['entropy']:
            processed = dict(processed_data)
            processed['required_losses'] = {
                'policy_gradient': False,
                'value': False,
                'pathwise': False,
                'entropy': True
            }
            _, _, entropy_loss, _, _, _, _ = self._compute_losses(mb_data, processed, epoch=0)
        
        if required_losses['pathwise']:
            processed = dict(processed_data)
            processed['required_losses'] = {
                'policy_gradient': False,
                'value': False,
                'pathwise': True,
                'entropy': False
            }
            _, _, _, pathwise_loss, _, _, _ = self._compute_losses(mb_data, processed, epoch=0)
        
        # Mirror HybridWrapper._analyze_gradients behavior
        if required_losses['pathwise'] and pathwise_loss.requires_grad:
            pathwise_loss.backward(retain_graph=True)
            self._record_cosine_gradients('pathwise')
        
        self.optimizer.zero_grad()
        
        if required_losses['policy_gradient'] and policy_loss.requires_grad:
            policy_loss.backward(retain_graph=True)
            self._record_cosine_gradients('cross_ppo')
        
        if processed_data.get('use_reinforce_loss', False) and reinforce_loss.requires_grad:
            reinforce_loss.backward(retain_graph=True)
            self._record_cosine_gradients('cross_reinforce')
        
        self.optimizer.zero_grad()
        
        if required_losses['value'] and value_loss.requires_grad:
            value_loss.backward(retain_graph=True)
        
        if required_losses['entropy'] and entropy_loss.requires_grad:
            entropy_loss.backward(retain_graph=True)
        
        self.optimizer.zero_grad()
    
    def _compute_old_discrete_log_probs(self, trajectory_data, processed_data):
        if ('discrete_logits' not in trajectory_data or trajectory_data['discrete_logits'] is None
                or 'discrete_action_indices' not in trajectory_data
                or trajectory_data['discrete_action_indices'] is None):
            return None
        
        effective_slice = processed_data['effective_slice']
        effective_T = processed_data['effective_T']
        B = processed_data['B']
        
        logits = trajectory_data['discrete_logits'][effective_slice]
        actions = trajectory_data['discrete_action_indices'][effective_slice]
        
        logits = logits.reshape(effective_T * B, *logits.shape[2:])
        actions = actions.reshape(effective_T * B, *actions.shape[2:])
        
        log_probs = self.model.get_discrete_log_probs(logits, actions)
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(-1)
        elif log_probs.dim() > 2:
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        return log_probs
    
    def _collect_grad_vector(self, loss, key, batch_size=None):
        self.optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        grad_vec = self._flatten_continuous_grads()
        self.optimizer.zero_grad()
        
        if grad_vec is None:
            return
        if batch_size is not None and self._shadow_alignment_batch_size is None:
            self._shadow_alignment_batch_size = int(batch_size)
        self._shadow_grad_buffers[key].append(grad_vec)
        self._shadow_norm_buffers[key].append(grad_vec.norm().item())
    
    def _flatten_continuous_grads(self):
        if not self._continuous_params:
            return None
        grads = []
        for _, param in self._continuous_params:
            if param.grad is None:
                grad = torch.zeros_like(param)
            else:
                grad = param.grad
            grads.append(grad.detach().flatten())
        return torch.cat(grads).cpu()
    
    def _compute_shadow_metrics(self):
        metrics = {}
        buffers = self._shadow_grad_buffers
        if not buffers:
            return metrics

        d_values, repeats = self._get_alignment_settings()
        if not d_values or repeats is None or self._shadow_alignment_batch_size is None:
            return metrics

        def add_cosine_metrics(prefix, vecs):
            if not vecs:
                return
            stacked = torch.stack(vecs)
            metrics.update(
                self._compute_loo_metrics_from_stacks(
                    stacked,
                    stacked,
                    self._shadow_alignment_batch_size,
                    d_values,
                    repeats,
                    f'continuous/{prefix}'
                )
            )

        add_cosine_metrics('continuous_ppo', buffers.get('continuous_ppo', []))
        add_cosine_metrics('continuous_reinforce', buffers.get('continuous_reinforce', []))
        add_cosine_metrics('pathwise_plus_cross_ppo', buffers.get('pathwise_plus_cross_ppo', []))
        add_cosine_metrics('pathwise_plus_cross_reinforce', buffers.get('pathwise_plus_cross_reinforce', []))

        vecs_continuous_ppo = buffers.get('continuous_ppo', [])
        if vecs_continuous_ppo:
            stacked = torch.stack(vecs_continuous_ppo)
            metrics.update(
                self._compute_alignment_from_stack(
                    stacked,
                    self._shadow_alignment_batch_size,
                    d_values,
                    repeats,
                    'continuous/continuous_ppo'
                )
            )

        vecs_continuous_reinforce = buffers.get('continuous_reinforce', [])
        vecs_pathwise_plus_cross_ppo = buffers.get('pathwise_plus_cross_ppo', [])
        vecs_pathwise_plus_cross_reinforce = buffers.get('pathwise_plus_cross_reinforce', [])

        if vecs_continuous_ppo and vecs_pathwise_plus_cross_ppo:
            n = min(len(vecs_continuous_ppo), len(vecs_pathwise_plus_cross_ppo))
            query_stack = torch.stack(vecs_continuous_ppo[:n])
            true_stack = torch.stack(vecs_pathwise_plus_cross_ppo[:n])
            metrics.update(
                self._compute_loo_metrics_from_stacks(
                    query_stack,
                    true_stack,
                    self._shadow_alignment_batch_size,
                    d_values,
                    repeats,
                    'continuous/continuous_ppo_vs_true_pathwise_plus_cross_ppo'
                )
            )

        if vecs_continuous_reinforce and vecs_pathwise_plus_cross_ppo:
            n = min(len(vecs_continuous_reinforce), len(vecs_pathwise_plus_cross_ppo))
            query_stack = torch.stack(vecs_continuous_reinforce[:n])
            true_stack = torch.stack(vecs_pathwise_plus_cross_ppo[:n])
            metrics.update(
                self._compute_loo_metrics_from_stacks(
                    query_stack,
                    true_stack,
                    self._shadow_alignment_batch_size,
                    d_values,
                    repeats,
                    'continuous/continuous_reinforce_vs_true_pathwise_plus_cross_ppo'
                )
            )

        if vecs_continuous_reinforce and vecs_pathwise_plus_cross_reinforce:
            n = min(len(vecs_continuous_reinforce), len(vecs_pathwise_plus_cross_reinforce))
            query_stack = torch.stack(vecs_continuous_reinforce[:n])
            true_stack = torch.stack(vecs_pathwise_plus_cross_reinforce[:n])
            metrics.update(
                self._compute_loo_metrics_from_stacks(
                    query_stack,
                    true_stack,
                    self._shadow_alignment_batch_size,
                    d_values,
                    repeats,
                    'continuous/continuous_reinforce_vs_true_pathwise_plus_cross_reinforce'
                )
            )
        
        # --- Restore True vs True Comparisons ---
        if vecs_continuous_ppo and vecs_pathwise_plus_cross_ppo:
            true_ppo = torch.stack(vecs_continuous_ppo).sum(dim=0)
            true_hybrid = torch.stack(vecs_pathwise_plus_cross_ppo).sum(dim=0)
            
            true_ppo_norm = true_ppo.norm() + 1e-8
            true_hybrid_norm = true_hybrid.norm() + 1e-8
            true_cosine = (true_ppo @ true_hybrid) / (true_ppo_norm * true_hybrid_norm)
            metrics['cosine/continuous/true_continuous_ppo_vs_true_pathwise_plus_cross_ppo'] = true_cosine.item()

        if vecs_continuous_reinforce and vecs_pathwise_plus_cross_reinforce:
            true_reinforce = torch.stack(vecs_continuous_reinforce).sum(dim=0)
            true_hybrid_reinforce = torch.stack(vecs_pathwise_plus_cross_reinforce).sum(dim=0)
            
            true_reinf_norm = true_reinforce.norm() + 1e-8
            true_hyb_reinf_norm = true_hybrid_reinforce.norm() + 1e-8
            true_cosine = (true_reinforce @ true_hybrid_reinforce) / (true_reinf_norm * true_hyb_reinf_norm)
            metrics['cosine/continuous/true_continuous_reinforce_vs_true_pathwise_plus_cross_reinforce'] = true_cosine.item()
        
        if vecs_continuous_ppo:
            metrics['shadow/continuous/num_batches'] = len(vecs_continuous_ppo)
        
        return metrics

