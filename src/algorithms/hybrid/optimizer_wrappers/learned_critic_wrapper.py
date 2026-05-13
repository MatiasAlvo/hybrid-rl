import torch
import torch.nn.functional as F

from src.algorithms.hybrid.optimizer_wrappers.hybrid_wrapper import HybridWrapper


class LearnedCriticWrapper(HybridWrapper):
    """
    HybridWrapper variant that replaces true pathwise loss with:
    - actor loss: -Q(s, pi(s))
    - critic loss: Bellman TD loss for Q
    while keeping PPO/value/entropy logic unchanged.
    """

    def __init__(
        self,
        model,
        optimizer,
        problem_params,
        device="cpu",
        ppo_params=None,
        base_lr=None,
        lr_multipliers=None,
    ):
        super().__init__(
            model,
            optimizer,
            problem_params,
            device=device,
            ppo_params=ppo_params,
            base_lr=base_lr,
            lr_multipliers=lr_multipliers,
        )
        self.gamma = float(getattr(model, "gamma", self.ppo_params.get("gamma", 0.99)))
        self.actor_warmup_epochs = int(self.ppo_params.get("actor_warmup_epochs", 20))
        self.learned_critic_epoch = 0
        self._warned_missing_next_observations = False
        self._actor_loss_values = []
        self._critic_loss_values = []
        self._q_values = []
        self._target_q_values = []

    def optimize(self, trajectory_data):
        self._actor_loss_values = []
        self._critic_loss_values = []
        self._q_values = []
        self._target_q_values = []
        metrics = super().optimize(trajectory_data)
        if self._actor_loss_values:
            metrics["loss/actor_critic"] = sum(self._actor_loss_values) / len(self._actor_loss_values)
        if self._critic_loss_values:
            metrics["loss/critic"] = sum(self._critic_loss_values) / len(self._critic_loss_values)
        if self._q_values:
            metrics["critic/q_mean"] = sum(self._q_values) / len(self._q_values)
        if self._target_q_values:
            metrics["critic/target_q_mean"] = sum(self._target_q_values) / len(self._target_q_values)
        return metrics

    def on_epoch_end(self):
        super().on_epoch_end()
        self.learned_critic_epoch += 1
        if hasattr(self.model, "soft_update_target"):
            self.model.soft_update_target()

    def _prepare_trajectory_data(self, trajectory_data):
        processed_data = super()._prepare_trajectory_data(trajectory_data)
        processed_data["critic_coef"] = self.ppo_params.get("critic_coef", 1.0)
        processed_data["actor_coef"] = self.ppo_params.get("actor_coef", processed_data["pathwise_coef"])
        return processed_data

    def _prepare_training_tensors(self, trajectory_data, advantages, returns, cum_rewards, processed_data):
        tensors = super()._prepare_training_tensors(
            trajectory_data, advantages, returns, cum_rewards, processed_data
        )
        effective_slice = processed_data["effective_slice"]
        effective_T = processed_data["effective_T"]
        B = processed_data["B"]

        if "next_observations" in trajectory_data and trajectory_data["next_observations"] is not None:
            tensors["next_observations"] = trajectory_data["next_observations"][effective_slice].reshape(
                effective_T * B, -1
            )
            rewards_for_critic = trajectory_data["rewards"][effective_slice]
            if self.ppo_params.get("reward_scaling", False):
                rewards_std = rewards_for_critic.std().detach()
                if rewards_std > 0:
                    rewards_for_critic = rewards_for_critic / (rewards_std + 1e-8)
            tensors["rewards_for_critic"] = rewards_for_critic.reshape(effective_T * B)
            terminated = trajectory_data["terminated"][effective_slice]
            if terminated.dim() == 1:
                terminated = terminated.unsqueeze(1).expand(-1, B)
            tensors["terminated_for_critic"] = terminated.reshape(effective_T * B)
        return tensors

    def _get_minibatch(self, tensors, mb_inds):
        mb_data = super()._get_minibatch(tensors, mb_inds)
        if "next_observations" in tensors:
            mb_data["next_observations"] = tensors["next_observations"][mb_inds]
        if "rewards_for_critic" in tensors:
            mb_data["rewards_for_critic"] = tensors["rewards_for_critic"][mb_inds]
        if "terminated_for_critic" in tensors:
            mb_data["terminated_for_critic"] = tensors["terminated_for_critic"][mb_inds]
        return mb_data

    def _optimization_step(self, policy_loss, value_loss, entropy_loss, pathwise_loss, processed_data):
        self.optimizer.zero_grad()
        required_losses = processed_data["required_losses"]
        loss = torch.tensor(0.0, device=self.device)

        if required_losses["policy_gradient"]:
            loss = loss + policy_loss
        if required_losses["value"]:
            loss = loss + value_loss
        if required_losses.get("learned_critic", False):
            loss = loss + pathwise_loss
        elif required_losses["pathwise"]:
            loss = loss + pathwise_loss
        if required_losses["entropy"]:
            loss = loss - entropy_loss

        grad_metrics = {}
        if loss.requires_grad and not processed_data.get("skip_update", False):
            loss.backward()
            if self.freeze_backbone:
                for name, param in self.model.named_parameters():
                    if ("backbone" in name or "discrete" in name) and param.grad is not None:
                        param.grad.zero_()
            if processed_data["max_grad_norm"] is not None:
                grad_metrics = self._clip_gradients_by_component(processed_data["max_grad_norm"])
            self.optimizer.step()
        return grad_metrics

    def _compute_losses(self, mb_data, processed_data, epoch, normalize_by_stores=False):
        policy_loss, value_loss, entropy_loss, pathwise_loss, metrics, raw_entropy, reinforce_loss = (
            super()._compute_losses(mb_data, processed_data, epoch, normalize_by_stores=normalize_by_stores)
        )

        required_losses = processed_data["required_losses"]
        if not required_losses.get("learned_critic", False):
            return policy_loss, value_loss, entropy_loss, pathwise_loss, metrics, raw_entropy, reinforce_loss

        if "next_observations" not in mb_data or "rewards_for_critic" not in mb_data:
            if not self._warned_missing_next_observations:
                print("Warning: trajectory_data missing next_observations; skipping learned critic losses.")
                self._warned_missing_next_observations = True
            return policy_loss, value_loss, entropy_loss, pathwise_loss, metrics, raw_entropy, reinforce_loss

        continuous_action = self._get_continuous_action_for_critic(
            mb_data["observations"], mb_data.get("discrete_action_indices")
        )

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.learned_critic_epoch >= self.actor_warmup_epochs:
            # DDPG-style actor update: gradient should flow through action, not into Q-network weights.
            q_requires_grad = []
            for p in self.model.q_net.parameters():
                q_requires_grad.append(p.requires_grad)
                p.requires_grad_(False)
            actor_loss = -self.model.q_net(mb_data["observations"], continuous_action).mean()
            actor_loss = processed_data["actor_coef"] * actor_loss
            for p, rg in zip(self.model.q_net.parameters(), q_requires_grad):
                p.requires_grad_(rg)

        with torch.no_grad():
            next_action = self._get_continuous_action_for_critic(mb_data["next_observations"])
            terminated = mb_data.get("terminated_for_critic")
            if terminated is None:
                not_done = 1.0
            else:
                not_done = 1.0 - terminated.float()
            target_q = mb_data["rewards_for_critic"] + self.gamma * not_done * self.model.q_net_target(
                mb_data["next_observations"], next_action
            )

        current_q = self.model.q_net(mb_data["observations"].detach(), continuous_action.detach())
        critic_loss = F.mse_loss(current_q, target_q)
        critic_loss = processed_data["critic_coef"] * critic_loss

        combined_loss = actor_loss + critic_loss
        self._actor_loss_values.append(float(actor_loss.item()))
        self._critic_loss_values.append(float(critic_loss.item()))
        self._q_values.append(float(current_q.mean().item()))
        self._target_q_values.append(float(target_q.mean().item()))

        return policy_loss, value_loss, entropy_loss, combined_loss, metrics, raw_entropy, reinforce_loss

    def _get_continuous_action_for_critic(self, observations, discrete_action_indices=None):
        raw_outputs = self.model.policy(observations, process_state=False)
        discrete_logits = raw_outputs.get("discrete")

        if discrete_action_indices is None:
            if discrete_logits is None:
                raise ValueError("Policy must provide discrete logits for learned critic action selection.")
            discrete_output = self.model.feature_registry.process_discrete_output(
                discrete_logits, argmax=True, sample=False, straight_through=False
            )
            discrete_action_indices = discrete_output["discrete_action_indices"]

        apply_scaling = not self.model.policy_config.get("disable_continuous_scaling", False)
        continuous_output = self.model.feature_registry.process_continuous_output(
            raw_outputs.get("continuous"),
            discrete_action_indices=discrete_action_indices,
            continuous_mean=raw_outputs.get("continuous_mean"),
            continuous_log_std=raw_outputs.get("continuous_log_std"),
            random_continuous=False,
            observations=None,
            apply_scaling=apply_scaling,
        )
        continuous_values = continuous_output["continuous_values"]
        if continuous_values is None:
            raise ValueError("Policy did not produce continuous outputs for learned critic.")
        if continuous_values.dim() == 2:
            return continuous_values
        if continuous_values.dim() != 3:
            raise ValueError(f"Expected continuous values with rank 2 or 3, got {continuous_values.dim()}.")

        idx = discrete_action_indices
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        if idx.dim() == 2 and idx.size(1) == 1:
            idx = idx.expand(-1, continuous_values.size(1))
        if idx.dim() != 2:
            raise ValueError("Discrete action indices must have shape [B] or [B, S].")
        if idx.size(1) != continuous_values.size(1):
            raise ValueError(
                f"Discrete indices store dim {idx.size(1)} does not match continuous store dim {continuous_values.size(1)}."
            )

        selected = continuous_values.gather(-1, idx.long().unsqueeze(-1)).squeeze(-1)
        return selected
