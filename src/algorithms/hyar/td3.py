# Adapted from HyAR (TJU-DRL-LAB/self-supervised-rl), Apache-2.0; see LICENSE and NOTICE.md here.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3_1 = nn.Linear(hidden_size, discrete_action_dim)
        self.l3_2 = nn.Linear(hidden_size, parameter_action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        discrete_action = self.max_action * torch.tanh(self.l3_1(a))
        parameter_action = self.max_action * torch.tanh(self.l3_2(a))
        return discrete_action, parameter_action


class Critic(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, layer_norm=False, hidden_size=256):
        super(Critic, self).__init__()
        # layer_norm: LayerNorm before each hidden ReLU, which keeps TD targets from diverging.
        self.layer_norm = layer_norm
        if layer_norm:
            self.n1, self.n2, self.n4, self.n5 = (nn.LayerNorm(hidden_size) for _ in range(4))

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + discrete_action_dim + parameter_action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def _hidden(self, x, layer, norm):
        x = layer(x)
        return F.relu(getattr(self, norm)(x) if self.layer_norm else x)

    def forward(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = self._hidden(sa, self.l1, "n1")
        q1 = self._hidden(q1, self.l2, "n2")
        q1 = self.l3(q1)

        q2 = self._hidden(sa, self.l4, "n4")
        q2 = self._hidden(q2, self.l5, "n5")
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = self._hidden(sa, self.l1, "n1")
        q1 = self._hidden(q1, self.l2, "n2")
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            discrete_action_dim,
            parameter_action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            reward_norm="welford",
            reward_norm_decay=0.999,
            fast_relabel=False,
            actor_lr=3e-4,
            critic_lr=3e-4,
            critic_layer_norm=False,
            target_q_max=None,
            hidden_size=256,
    ):
        self.discrete_action_dim = discrete_action_dim
        self.parameter_action_dim = parameter_action_dim

        self.action_max = torch.from_numpy(np.ones((self.discrete_action_dim,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()

        self.action_parameter_max = torch.from_numpy(np.ones((self.parameter_action_dim,))).float().to(device)
        self.action_parameter_min = -self.action_parameter_max.detach()
        # print(" self.action_parameter_max_numpy", self.action_parameter_max)
        self.action_parameter_range = (self.action_parameter_max - self.action_parameter_min)

        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action,
                           hidden_size=hidden_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim,
                             layer_norm=critic_layer_norm, hidden_size=hidden_size).to(device)
        # target_q_max: upper bound on TD targets. With nonpositive rewards (costs) every true
        # Q-value is <= 0, so 0 is a safe bound that stops over-estimation from compounding.
        self.target_q_max = target_q_max
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        # Running reward standardization (adaptive; mirrors HPO's running batch-std cost
        # normalization, paper C.2). S-LQR costs span 4+ orders of magnitude as the policy
        # improves (random ~1e4 -> optimal ~1), so a FIXED scale cannot work; TD3 critic targets
        # need O(1) rewards. We track a running mean/std over sampled-batch rewards and divide the
        # reward by (std + eps) when forming target_Q. Enabled via normalize_reward.
        # reward_norm: "welford" divides by the std of every reward sampled since the start (the
        # original port), "ema" by an exponential moving std that forgets early random-policy
        # costs, "none" leaves rewards unscaled.
        if reward_norm not in ("welford", "ema", "none"):
            raise ValueError(f"reward_norm must be welford, ema or none, got {reward_norm!r}")
        self.reward_norm = reward_norm
        self.normalize_reward = reward_norm != "none"
        self.reward_norm_decay = float(reward_norm_decay)
        self._rew_count = 0.0
        self._rew_mean = 0.0
        self._rew_M2 = 0.0  # for Welford running variance
        self._ema_steps = 0
        self._ema_mean = torch.zeros((), device=device)
        self._ema_sq = torch.zeros((), device=device)
        # fast_relabel keeps the relabeling step on the device instead of round-tripping
        # through numpy; same computation, but not bit-identical to the upstream code path.
        self.fast_relabel = fast_relabel

    def _update_reward_stats(self, reward_batch):
        """Running update of the reward scale over a batch of rewards (flattened)."""
        if self.reward_norm == "ema":
            r = reward_batch.detach().reshape(-1)
            d = self.reward_norm_decay
            self._ema_steps += 1
            self._ema_mean = d * self._ema_mean + (1 - d) * r.mean()
            self._ema_sq = d * self._ema_sq + (1 - d) * (r * r).mean()
            return
        r = reward_batch.detach().reshape(-1)
        n_b = r.numel()
        if n_b == 0:
            return
        batch_mean = float(r.mean().item())
        batch_var = float(r.var(unbiased=False).item())
        # combine (Chan's parallel algorithm)
        delta = batch_mean - self._rew_mean
        tot = self._rew_count + n_b
        self._rew_mean += delta * n_b / tot
        self._rew_M2 += batch_var * n_b + (delta * delta) * self._rew_count * n_b / tot
        self._rew_count = tot

    def _reward_std(self):
        if self.reward_norm == "ema":
            if self._ema_steps == 0:
                return 1.0
            correction = 1 - self.reward_norm_decay ** self._ema_steps
            mean = self._ema_mean / correction
            var = (self._ema_sq / correction - mean * mean).clamp(min=1e-12)
            return var.sqrt().clamp(min=1e-6)
        if self._rew_count < 2:
            return 1.0
        var = self._rew_M2 / self._rew_count
        return max(1e-6, var ** 0.5)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        all_discrete_action, all_parameter_action = self.actor(state)
        return all_discrete_action.cpu().data.numpy().flatten(), all_parameter_action.cpu().data.numpy().flatten()

    def _record_stats(self, **values):
        """Accumulate training statistics as tensors (no device sync until pop_stats)."""
        stats = self.__dict__.setdefault("_stats", {})
        for k, v in values.items():
            stats[k] = stats.get(k, 0.0) + v.detach().float()
        self._stats_n = getattr(self, "_stats_n", 0) + 1

    def pop_stats(self):
        n = getattr(self, "_stats_n", 0)
        out = {k: float(v) / n for k, v in getattr(self, "_stats", {}).items()} if n else {}
        self._stats, self._stats_n = {}, 0
        return out

    def _c_rate_tensors(self, c_rate, like):
        """Latent-space-constraint bounds as (median, offset) tensors, rebuilt when c_rate changes."""
        if getattr(self, "_c_rate_key", None) is not c_rate:
            c = torch.as_tensor(np.asarray(c_rate, dtype=np.float32), device=like.device)
            median = (c[:, 0] - c[:, 1]) / 2
            self._c_rate_cache = (median, c[:, 0] - median)
            self._c_rate_key = c_rate
        return self._c_rate_cache

    def _relabel_on_device(self, action_rep, c_rate, recon_s_rate, state, discrete_action, parameter_action,
                           discrete_emb, parameter_emb, state_next_state):
        """The upstream relabeling step (representation shift correction) without numpy round trips."""
        with torch.no_grad():
            discrete_emb_ = action_rep.get_embedding(discrete_action.reshape(-1).long())
            noise_discrete = (torch.randn_like(discrete_emb_) * 0.1).clamp(-self.noise_clip, self.noise_clip)
            discrete_emb_table = discrete_emb_.clamp(-self.max_action, self.max_action)
            discrete_emb_table_noise = (discrete_emb_ + noise_discrete).clamp(-self.max_action, self.max_action)

            discrete_action_old = action_rep.get_match_scores(discrete_emb).argmax(dim=1, keepdim=True)
            d_bing = (discrete_action == discrete_action_old).float()
            discrete_emb_ = d_bing * discrete_emb + (1.0 - d_bing) * discrete_emb_table_noise

            _, predict_delta_state = action_rep.vae.decode(state, parameter_emb, discrete_emb_table)
            delta_state = (predict_delta_state - state_next_state).pow(2).mean(dim=1, keepdim=True)
            s_bing = (delta_state.abs() < recon_s_rate).float()

            _, _, mean, std = action_rep.vae(state, discrete_emb_table, parameter_action)
            parameter_emb_ = mean + std * torch.randn_like(std)
            median, offset = self._c_rate_tensors(c_rate, parameter_emb_)
            parameter_emb_ = (parameter_emb_ - offset) / median
            parameter_emb_ = s_bing * parameter_emb + (1 - s_bing) * parameter_emb_

            return (discrete_emb_.clamp(-self.max_action, self.max_action),
                    parameter_emb_.clamp(-self.max_action, self.max_action),
                    d_bing.mean(), s_bing.mean())

    def train(self, replay_buffer, action_rep, c_rate, recon_s_rate, batch_size=256):
        recon_s_rate = recon_s_rate * 5.0
        self.total_it += 1
        # Sample replay buffer
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        # print("discrete_emb----------",discrete_emb)
        if self.fast_relabel:
            discrete_emb, parameter_emb, discrete_relable_rate, parameter_relable_rate = self._relabel_on_device(
                action_rep, c_rate, recon_s_rate, state, discrete_action, parameter_action,
                discrete_emb, parameter_emb, state_next_state)
        else:
            with torch.no_grad():

                discrete_emb_ = action_rep.get_embedding(discrete_action.reshape(1,
                                                                                 -1).squeeze().long()).to(device)
                # discrete relable need noise
                noise_discrete = (
                        torch.randn_like(discrete_emb_) * 0.1
                ).clamp(-self.noise_clip, self.noise_clip)
                discrete_emb_table = discrete_emb_.clamp(-self.max_action, self.max_action)
                discrete_emb_table_noise = (discrete_emb_ + noise_discrete).clamp(-self.max_action, self.max_action)

                discrete_action_old = action_rep.select_discrete_action(discrete_emb).reshape(-1, 1)
                d_new = discrete_action.cpu().numpy()
                d_old = discrete_action_old
                d_bing = (d_new == d_old) * 1
                # discrete_relable_rate
                discrete_relable_rate = sum(d_bing.reshape(1, -1)[0]) / batch_size
                d_bing = torch.FloatTensor(d_bing).float().to(device)
                discrete_emb_ = d_bing * discrete_emb + (1.0 - d_bing) * discrete_emb_table_noise
                # print("discrete_emb_final",discrete_emb_)

                predict_delta_state = action_rep.select_delta_state(state, parameter_emb, discrete_emb_table)

                # print("predict_delta_state",predict_delta_state)
                # print("state_next_state",state_next_state.cpu().numpy())
                delta_state = (np.square(predict_delta_state - state_next_state.cpu().numpy())).mean(axis=1).reshape(-1, 1)
                # delta_state=predict_delta_state-state_next_state.cpu().numpy()
                # delta_state=np.mean(delta_state, axis=1).reshape(-1, 1)
                s_bing = (abs(delta_state) < recon_s_rate) * 1
                parameter_relable_rate = sum(s_bing.reshape(1, -1)[0]) / batch_size
                s_bing = torch.FloatTensor(s_bing).float().to(device)
                # print("s_bing",s_bing)

                recon_c, recon_s, mean, std = action_rep.vae(state, discrete_emb_table, parameter_action)
                parameter_emb_ = mean + std * torch.randn_like(std)
                for i in range(len(parameter_emb_[0])):
                    parameter_emb_[:, i:i + 1] = self.true_parameter_emb(parameter_emb_[:, i:i + 1], c_rate, i)
                # print("parameter_emb",parameter_emb)
                # print("parameter_emb_",parameter_emb_)

                parameter_emb_ = s_bing * parameter_emb + (1 - s_bing) * parameter_emb_
                # print("parameter_emb_final",parameter_emb_)

                discrete_emb_ = discrete_emb_.clamp(-self.max_action, self.max_action)
                parameter_emb_ = parameter_emb_.clamp(-self.max_action, self.max_action)

                discrete_emb = discrete_emb_
                parameter_emb = parameter_emb_

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise_discrete = (
                    torch.randn_like(discrete_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            noise_parameter = (
                    torch.randn_like(parameter_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_discrete_action, next_parameter_action = self.actor_target(next_state)
            next_discrete_action = (next_discrete_action + noise_discrete).clamp(-self.max_action, self.max_action)
            next_parameter_action = (next_parameter_action + noise_parameter).clamp(-self.max_action, self.max_action)

            # Compute the target Q value. Reward is standardized by a running std (adaptive to
            # the policy's current cost scale) so critic targets stay O(1) as costs shrink from
            # ~1e4 (random) to ~1 (optimal). Fair analog of HPO's running cost standardization.
            if self.normalize_reward:
                self._update_reward_stats(reward)
                reward_n = reward / self._reward_std()
            else:
                reward_n = reward
            target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_action, next_parameter_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_n + not_done * self.discount * target_Q
            if self.target_q_max is not None:
                target_Q = target_Q.clamp(max=self.target_q_max)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, discrete_emb, parameter_emb)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if getattr(self, "track_stats", False):
            self._record_stats(critic_loss=critic_loss, q=current_Q1.mean(), target_q=target_Q.mean(),
                               reward_scale=torch.as_tensor(self._reward_std() if self.normalize_reward else 1.0))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            inverting_gradients = True
            # inverting_gradients = False
            # Compute actor loss
            if inverting_gradients:
                with torch.no_grad():
                    next_discrete_action, next_parameter_action = self.actor(state)
                    action_params = torch.cat((next_discrete_action, next_parameter_action), dim=1)
                action_params.requires_grad = True
                actor_loss = self.critic.Q1(state, action_params[:, :self.discrete_action_dim],
                                            action_params[:, self.discrete_action_dim:]).mean()
            else:
                next_discrete_action, next_parameter_action = self.actor(state)
                actor_loss = -self.critic.Q1(state, next_discrete_action, next_parameter_action).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if inverting_gradients:
                from copy import deepcopy
                delta_a = deepcopy(action_params.grad.data)
                # 2 - apply inverting gradients and combine with gradients from actor
                actions, action_params = self.actor(Variable(state))
                action_params = torch.cat((actions, action_params), dim=1)
                delta_a[:, self.discrete_action_dim:] = self._invert_gradients(
                    delta_a[:, self.discrete_action_dim:].cpu(),
                    action_params[:, self.discrete_action_dim:].cpu(),
                    grad_type="action_parameters", inplace=True)
                delta_a[:, :self.discrete_action_dim] = self._invert_gradients(
                    delta_a[:, :self.discrete_action_dim].cpu(),
                    action_params[:, :self.discrete_action_dim].cpu(),
                    grad_type="actions", inplace=True)
                out = -torch.mul(delta_a, action_params)
                self.actor.zero_grad()
                out.backward(torch.ones(out.shape).to(device))
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)

            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return discrete_relable_rate, parameter_relable_rate


    # def save(self, filename, directory):
    #     torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
    #     # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))
    #
    # def load(self, filename, directory):
    #     self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
    #     # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)


    def save(self, filename,directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename,directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):

        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # Same element-wise products as upstream's loop over rows, computed in one pass.
            index = grad > 0
            grad.copy_(torch.where(index,
                                   grad * ((max_p - vals) / rnge),
                                   grad * ((vals - min_p) / rnge)))

        return grad

    def count_boundary(self, c_rate):
        median = (c_rate[0] - c_rate[1]) / 2
        offset = c_rate[0] - 1 * median
        return median, offset

    def true_parameter_emb(self, parameter_action, c_rate, i):
        # parameter_action_ = parameter_action.clone()
        median, offset = self.count_boundary(c_rate[i])
        # parameter_action_[i] = parameter_action_[i] * median + offset
        parameter_action = (parameter_action - offset) / median
        return parameter_action
