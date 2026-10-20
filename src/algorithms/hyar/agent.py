"""HyARAgent -- thin wrapper around the ported HyAR-TD3 pieces.

Wires together the three ported modules:
  * embedding.ActionRepresentation_vae.Action_representation  (discrete embedding
    table + conditional VAE + delta-state head + LSC via get_c_rate/z_range)
  * agents.P_TD3_relable.TD3  (TD3 in latent space + inverting gradients + RSC relabel)
  * utils.ReplayBuffer  (dual-store: raw hybrid action AND its latent embedding)

It consumes an environment satisfying HyAR/bridge/CONTRACT.md:
  attributes: state_dim (int), K (int, #discrete actions), param_dim (int, cont. dim)
  methods:    reset(scenario_idx=None) -> state
              step(k:int, x:np.ndarray) -> (next_state, reward, done, info)

HyAR dimensional mapping (CONTRACT -> reference names):
  state_dim            = env.state_dim
  discrete_action_dim  = env.K            (rows of the embedding table; nearest-neighbour decode -> k)
  parameter_action_dim = env.param_dim    (VAE reconstructs the full x vector; x is GLOBAL, shared
                                           across discrete actions -- unlike gym-Platform's per-action
                                           split, so all_parameter_action_dim == param_dim too)
  reduced_action_dim   (discrete embedding dim, kept modest, default 6) == TD3 discrete-latent dim
  reduce_parameter_action_dim (VAE latent dim, in the z_range-supported set {3,4,6,8,10,12,16},
                               default 6) == TD3 continuous-latent dim

The continuous param produced by select_action is the raw VAE decode output, i.e. in
(-max_action, max_action) == (-1, 1); the env adapter is responsible for mapping it to the real
action range (softplus for JRP order-qty, identity for LQR control). See report notes.
"""

import copy
import numpy as np
import torch

from src.algorithms.hyar import action_representation as ActionRepresentation_vae
from src.algorithms.hyar import td3 as P_TD3_relable
from src.algorithms.hyar import replay_buffer as hyar_utils


def _get_device():
    # Respects CUDA_VISIBLE_DEVICES: cuda -> the (single) visible device, else cpu.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_boundary(c_rate_i, eps=1e-6):
    """Latent-Space-Constraint boundary for one latent dim.

    c_rate_i = [up, down] (up >= down). median maps [-1,1] -> half-range,
    offset = midpoint. Guarded so a degenerate (up==down) dim does not divide by zero.
    """
    median = (c_rate_i[0] - c_rate_i[1]) / 2.0
    offset = c_rate_i[0] - 1.0 * median
    if abs(median) < eps:
        median = eps if median >= 0 else -eps
    return median, offset


def true_parameter_action(parameter_emb, c_rate):
    """LSC rescale: map a latent vector in [-1,1]^L to the constrained z-range [down,up]^L.

    Mirrors main_embedding_platform_td3.true_parameter_action, with a divide-by-zero guard.
    """
    out = copy.deepcopy(np.asarray(parameter_emb, dtype=np.float32))
    for i in range(len(out)):
        median, offset = _count_boundary(c_rate[i])
        out[i] = out[i] * median + offset
    return out


class HyARAgent(object):
    def __init__(
        self,
        state_dim,
        K,
        param_dim,
        reduced_action_dim=6,          # discrete embedding dim (modest, 4-6)
        reduce_parameter_action_dim=6, # VAE latent dim; must be in {3,4,6,8,10,12,16}
        max_action=1.0,
        delta_scale=None,              # delta-state head output scale; None -> auto from warmup deltas
        vae_hidden_size=256,           # VAE MLP width; raise for high-dim dynamics (robustness check)
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.5,
        policy_freq=2,
        batch_size=128,
        embed_lr=1e-4,
        expl_noise_initial=1.0,
        expl_noise=0.1,
        epsilon_steps=1000,
        c_rate_batch=5000,
        c_rate_range=2,
        buffer_size=int(1e6),
        embedding_buffer_size=int(1e6),
        seed=None,
        buffer_device=None,            # None: numpy buffers (original); a device: float32 tensor buffers there
        reward_norm="welford",         # TD3 reward scaling: welford (original), ema or none
        reward_norm_decay=0.999,
        fast_relabel=False,            # relabeling on the device instead of via numpy
        vae_kl_weight=0.5,             # weight of the VAE's KL term (upstream: 0.5)
        vae_persistent_optimizer=False,  # keep one Adam across VAE steps (upstream rebuilds it each step)
        actor_lr=3e-4,
        critic_lr=3e-4,
        critic_layer_norm=False,       # LayerNorm in the critic's hidden layers
        target_q_max=None,             # upper bound on TD targets (0 is valid when rewards are costs)
        td3_hidden_size=256,           # width of the TD3 actor/critic hidden layers (upstream: 256)
    ):
        if int(reduce_parameter_action_dim) < 1:
            raise ValueError(f"reduce_parameter_action_dim must be >= 1, got {reduce_parameter_action_dim}")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = _get_device()
        self.state_dim = int(state_dim)
        self.K = int(K)
        self.param_dim = int(param_dim)
        self.reduced_action_dim = int(reduced_action_dim)       # discrete latent dim
        self.parameter_emb_dim = int(reduce_parameter_action_dim)  # continuous latent dim
        self.max_action = float(max_action)
        self.batch_size = int(batch_size)
        self.embed_lr = float(embed_lr)
        self.vae_hidden_size = int(vae_hidden_size)
        self.vae_kl_weight = float(vae_kl_weight)
        self.vae_persistent_optimizer = bool(vae_persistent_optimizer)
        # delta_scale: None -> auto-estimate from warmup state-deltas (recommended for unstable
        # dynamics like LQR where normalized deltas exceed the tanh(±1) delta-head range). A
        # float pins it. 1.0 reproduces original HyAR. Stored; VAE (re)built with it in warmup.
        self._delta_scale_req = delta_scale
        self.delta_scale = 1.0 if delta_scale is None else float(delta_scale)

        # exploration schedule (per explore-select call)
        self.expl_noise_initial = float(expl_noise_initial)
        self.expl_noise = float(expl_noise)
        self.epsilon_steps = int(epsilon_steps)
        self._explore_calls = 0

        # LSC sampling
        self.c_rate_batch = int(c_rate_batch)
        self.c_rate_range = int(c_rate_range)

        # ----- action representation (VAE + discrete embedding table) -----
        self._build_action_rep(self.delta_scale)

        # ----- TD3 policy operating purely in latent space -----
        self.policy = P_TD3_relable.TD3(
            state_dim=self.state_dim,
            discrete_action_dim=self.reduced_action_dim,
            parameter_action_dim=self.parameter_emb_dim,
            max_action=self.max_action,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise * self.max_action,
            noise_clip=noise_clip * self.max_action,
            policy_freq=policy_freq,
            reward_norm=reward_norm,
            reward_norm_decay=reward_norm_decay,
            fast_relabel=fast_relabel,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            critic_layer_norm=critic_layer_norm,
            target_q_max=target_q_max,
            hidden_size=td3_hidden_size,
        )

        # ----- dual replay buffers -----
        def make_buffer(**kw):
            if buffer_device is None:
                return hyar_utils.ReplayBuffer(**kw)
            return hyar_utils.TorchReplayBuffer(device=buffer_device, **kw)

        # main buffer: stores raw (k, x) AND the latent embeddings used -> TD3 training
        self.replay_buffer = make_buffer(
            state_dim=self.state_dim,
            discrete_action_dim=1,
            parameter_action_dim=self.param_dim,
            all_parameter_action_dim=self.param_dim,
            discrete_emb_dim=self.reduced_action_dim,
            parameter_emb_dim=self.parameter_emb_dim,
            max_size=int(buffer_size),
        )
        # embedding buffer: only (state, k, x, state_next_state) matter -> VAE training
        self.replay_buffer_embedding = make_buffer(
            state_dim=self.state_dim,
            discrete_action_dim=1,
            parameter_action_dim=self.param_dim,
            all_parameter_action_dim=self.param_dim,
            discrete_emb_dim=self.reduced_action_dim,
            parameter_emb_dim=self.parameter_emb_dim,
            max_size=int(embedding_buffer_size),
        )

        # LSC state (set by pretrain / refresh)
        self.c_rate = None
        self.recon_s = None

        # cache of the latents used by the most recent select_action, for store_last_transition
        self._pending = None

    # ------------------------------------------------------------------ #
    #  Warm-up random rollouts + VAE pre-training                         #
    # ------------------------------------------------------------------ #
    def _random_action(self):
        k = int(np.random.randint(0, self.K))
        x = np.random.uniform(-self.max_action, self.max_action, size=self.param_dim).astype(np.float32)
        return k, x

    def _build_action_rep(self, delta_scale):
        """(Re)build the VAE + embedding table with a given delta-state output scale."""
        self.delta_scale = float(delta_scale)
        self.action_rep = ActionRepresentation_vae.Action_representation(
            state_dim=self.state_dim,
            action_dim=self.K,
            parameter_action_dim=self.param_dim,
            reduced_action_dim=self.reduced_action_dim,
            reduce_parameter_action_dim=self.parameter_emb_dim,
            embed_lr=self.embed_lr,
            vae_hidden_size=self.vae_hidden_size,
            delta_scale=self.delta_scale,
            kl_weight=self.vae_kl_weight,
            persistent_optimizer=self.vae_persistent_optimizer,
        )

    def _auto_delta_scale(self):
        """Estimate delta-state output scale from warmup deltas so the tanh head isn't clipped.

        Uses the 99th percentile of |state_next_state| over the embedding buffer; the delta head
        outputs delta_scale*tanh(.), so ~99% of targets land inside the representable range.
        Clamped to >=1.0 (never shrink below original HyAR)."""
        n = min(self.replay_buffer_embedding.size, 20000)
        if n <= 0:
            return 1.0
        d = self.replay_buffer_embedding.state_next_state[:n]
        d = np.abs(d.cpu().numpy() if torch.is_tensor(d) else d)
        p99 = float(np.percentile(d, 99))
        return max(1.0, round(p99, 3))

    def warmup_and_pretrain_vae(self, env, n_warmup_eps=1000, n_vae_steps=5000,
                                vae_batch_size=64, verbose=True):
        """Fill the embedding buffer with random rollouts, then pre-train the VAE."""
        # ---- random warm-up rollouts (embedding buffer only) ----
        total_transitions = 0
        for ep in range(int(n_warmup_eps)):
            state = np.asarray(env.reset(), dtype=np.float32)
            done = False
            while not done:
                k, x = self._random_action()
                next_state, reward, done, _ = env.step(k, x)
                next_state = np.asarray(next_state, dtype=np.float32)
                state_next_state = next_state - state
                self.replay_buffer_embedding.add(
                    state, discrete_action=k, parameter_action=x,
                    all_parameter_action=x,
                    discrete_emb=np.zeros(self.reduced_action_dim, dtype=np.float32),
                    parameter_emb=np.zeros(self.parameter_emb_dim, dtype=np.float32),
                    next_state=next_state, state_next_state=state_next_state,
                    reward=reward, done=done)
                state = next_state
                total_transitions += 1
        if verbose:
            print(f"[warmup] collected {total_transitions} random transitions "
                  f"over {n_warmup_eps} episodes.")

        # ---- auto delta_scale: fit the tanh delta-state head to the observed delta range ----
        if self._delta_scale_req is None:
            est = self._auto_delta_scale()
            if abs(est - self.delta_scale) > 1e-9:
                self._build_action_rep(est)  # rebuild VAE with fitted scale (pre-training)
            if verbose:
                print(f"[delta_scale] auto-estimated = {self.delta_scale:.3f} "
                      f"(delta-state head now spans +/-{self.delta_scale:.2f})")

        # ---- VAE pre-training ----
        self.c_rate, self.recon_s = self._vae_train(
            train_step=int(n_vae_steps), batch_size=int(vae_batch_size), verbose=verbose)
        if verbose:
            print(f"[vae-pretrain] done. recon_s={float(self.recon_s):.5f}")
        return self.c_rate, self.recon_s

    def warmup_batched_and_pretrain_vae(self, wrapper, n_batched_eps=5, n_vae_steps=5000,
                                        vae_batch_size=64, verbose=True):
        """Like warmup_and_pretrain_vae, but each warm-up episode runs every training scenario at once.

        Upstream HyAR collects 5000 random episodes before pretraining the VAE; five batched
        episodes over the 1024 training scenarios give a comparable amount of data.
        """
        total_transitions = 0
        for _ in range(int(n_batched_eps)):
            raw = wrapper._train_batch_reset()
            z = wrapper._norm_state_torch(raw).detach().cpu().numpy().astype(np.float32)
            B = z.shape[0]
            done = False
            while not done:
                k = np.random.randint(0, self.K, size=B)
                x = np.random.uniform(-self.max_action, self.max_action,
                                      size=(B, self.param_dim)).astype(np.float32)
                raw_next, reward, done, _ = wrapper._train_batch_step(k, x)
                z_next = wrapper._norm_state_torch(raw_next).detach().cpu().numpy().astype(np.float32)
                self.replay_buffer_embedding.add_batch(
                    z, k, x, x, np.zeros((B, self.reduced_action_dim), dtype=np.float32),
                    np.zeros((B, self.parameter_emb_dim), dtype=np.float32),
                    z_next, z_next - z, reward, np.full(B, float(done)))
                z = z_next
                total_transitions += B
        if verbose:
            print(f"[warmup] collected {total_transitions} random transitions over "
                  f"{n_batched_eps} batched episodes.")
        if self._delta_scale_req is None:
            est = self._auto_delta_scale()
            if abs(est - self.delta_scale) > 1e-9:
                self._build_action_rep(est)
            if verbose:
                print(f"[delta_scale] auto-estimated = {self.delta_scale:.3f}")
        self.c_rate, self.recon_s = self._vae_train(
            train_step=int(n_vae_steps), batch_size=int(vae_batch_size), verbose=verbose)
        if verbose:
            print(f"[vae-pretrain] done. recon_s={float(self.recon_s):.5f}")
        return self.c_rate, self.recon_s

    def _vae_train(self, train_step, batch_size, verbose=False):
        """Run `train_step` VAE grad steps on the embedding buffer, then recompute LSC."""
        initial_losses = []
        for counter in range(int(train_step) + 10):
            (state, discrete_action, parameter_action, all_parameter_action,
             discrete_emb, parameter_emb, next_state, state_next_state,
             reward, not_done) = self.replay_buffer_embedding.sample(batch_size)

            vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.action_rep.unsupervised_loss(
                state,
                discrete_action.reshape(1, -1).squeeze().long(),
                parameter_action,
                state_next_state,
                batch_size, self.embed_lr)
            initial_losses.append(np.mean([vae_loss]))

            if verbose and counter % 1000 == 0 and counter >= 1000:
                print(f"[vae] step {counter} loss(mean50)={np.mean(initial_losses[-50:]):.5f} "
                      f"(s={float(recon_loss_s):.4f} c={float(recon_loss_c):.4f} KL={float(KL_loss):.4f})")

            # converge early once representations stabilise
            if (len(initial_losses) >= train_step
                    and np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:])):
                break

        return self._compute_c_rate(batch_size=self.c_rate_batch, range_rate=self.c_rate_range)

    def _compute_c_rate(self, batch_size, range_rate):
        n = max(1, min(batch_size, self.replay_buffer_embedding.size))
        (state, discrete_action, parameter_action, all_parameter_action,
         discrete_emb, parameter_emb, next_state, state_next_state,
         reward, not_done) = self.replay_buffer_embedding.sample(n)
        c_rate, recon_s = self.action_rep.get_c_rate(
            state, discrete_action.reshape(1, -1).squeeze().long(), parameter_action,
            state_next_state, batch_size=n, range_rate=range_rate)
        return c_rate, recon_s

    def refresh_vae(self, train_step=1, batch_size=64):
        """Periodic VAE refresh during the TD3 loop (Representation-Shift-Correction upkeep)."""
        self.c_rate, self.recon_s = self._vae_train(
            train_step=int(train_step), batch_size=int(batch_size), verbose=False)
        return self.c_rate, self.recon_s

    # ------------------------------------------------------------------ #
    #  Action selection: latent -> (k, x)                                #
    # ------------------------------------------------------------------ #
    def _current_epsilon(self):
        t = self._explore_calls
        if t < self.epsilon_steps:
            return self.expl_noise_initial - (self.expl_noise_initial - self.expl_noise) * (t / self.epsilon_steps)
        return self.expl_noise

    def select_action(self, state, explore=False):
        """state -> (k in [0,K), x with shape (param_dim,)).

        Does: TD3 actor -> latents; (optional) exploration noise; LSC rescale of the
        continuous latent; nearest-neighbour decode of the discrete latent -> k;
        VAE decode(state, rescaled_latent, quantised_discrete_emb) -> continuous x.
        Caches the (pre-rescale, post-noise) latents + (k, x) for store_last_transition.
        """
        if self.c_rate is None:
            raise RuntimeError("LSC c_rate is not set -- call warmup_and_pretrain_vae() first.")

        state = np.asarray(state, dtype=np.float32)
        discrete_emb, parameter_emb = self.policy.select_action(state)  # latents in [-1,1]

        if explore:
            eps = self._current_epsilon()
            discrete_emb = (discrete_emb + np.random.normal(
                0, self.max_action * eps, size=self.reduced_action_dim)).clip(-self.max_action, self.max_action)
            parameter_emb = (parameter_emb + np.random.normal(
                0, self.max_action * eps, size=self.parameter_emb_dim)).clip(-self.max_action, self.max_action)
            self._explore_calls += 1

        # LSC rescale of the continuous latent into the constrained z-range
        true_parameter_emb = true_parameter_action(parameter_emb, self.c_rate)

        # discrete: nearest-neighbour in the embedding table -> k
        discrete_action_embedding = torch.from_numpy(
            copy.deepcopy(discrete_emb)).float().reshape(1, -1)
        k = self.action_rep.select_discrete_action(discrete_action_embedding)
        k = int(k)
        # quantised (tanh'd table) embedding for the chosen k -> conditions the VAE decode
        discrete_emb_1 = self.action_rep.get_embedding(k).cpu().view(-1).data.numpy()

        # continuous: VAE decode -> x (param_dim,), in (-max_action, max_action)
        x = self.action_rep.select_parameter_action(state, true_parameter_emb, discrete_emb_1)
        x = np.asarray(x, dtype=np.float32).reshape(self.param_dim)

        # cache latents (pre-rescale, post-noise) for buffer storage
        self._pending = {
            "discrete_emb": np.asarray(discrete_emb, dtype=np.float32),
            "parameter_emb": np.asarray(parameter_emb, dtype=np.float32),
            "discrete_action": k,
            "parameter_action": x,
        }
        return k, x

    # ------------------------------------------------------------------ #
    #  BATCHED action selection (vectorized over B states) -- for the      #
    #  scenario-batched rollout/eval. Semantically identical to calling    #
    #  select_action per row, but runs the actor / LSC / NN-decode / VAE    #
    #  decode on the whole batch at once. Returns (k[B], x[B,param_dim])    #
    #  and, when explore, caches per-row latents for store_batch_transition.#
    # ------------------------------------------------------------------ #
    def _true_parameter_emb_batch(self, parameter_emb_bt):
        """LSC rescale for a batch: [B, L] latent in [-1,1] -> constrained z-range, per dim.

        Mirrors true_parameter_action()/count_boundary but vectorized. c_rate[i]=[up,down].
        """
        out = parameter_emb_bt.clone()
        eps = 1e-6
        for i in range(out.shape[1]):
            up, down = float(self.c_rate[i][0]), float(self.c_rate[i][1])
            median = (up - down) / 2.0
            offset = up - median
            if abs(median) < eps:
                median = eps if median >= 0 else -eps
            out[:, i] = out[:, i] * median + offset
        return out

    @torch.no_grad()
    def select_action_batch(self, state_bt, explore=False):
        """Vectorized select_action. state_bt: np.ndarray or torch [B, state_dim].

        Returns (k: np.int64[B], x: np.float32[B, param_dim]). When explore=True, caches
        per-row latents (post-noise, pre-rescale) in self._pending_batch for
        store_batch_transition().
        """
        if self.c_rate is None:
            raise RuntimeError("LSC c_rate is not set -- call warmup_and_pretrain_vae() first.")
        st = torch.as_tensor(state_bt, dtype=torch.float32, device=self.device)
        B = st.shape[0]

        # actor -> latents in [-max_action, max_action]
        discrete_emb, parameter_emb = self.policy.actor(st)  # [B, dsc], [B, L]

        if explore:
            eps = self._current_epsilon()
            sd = self.max_action * eps
            discrete_emb = (discrete_emb + torch.randn_like(discrete_emb) * sd).clamp(
                -self.max_action, self.max_action)
            parameter_emb = (parameter_emb + torch.randn_like(parameter_emb) * sd).clamp(
                -self.max_action, self.max_action)
            self._explore_calls += 1  # one schedule tick per batched step (see note in driver)

        # LSC rescale of the continuous latent
        true_parameter_emb = self._true_parameter_emb_batch(parameter_emb)  # [B, L]

        # discrete: nearest-neighbour in the (tanh'd) embedding table -> k[B]
        # get_match_scores handles a [B, dsc] query and returns [B, K] similarities.
        sims = self.action_rep.get_match_scores(discrete_emb)  # [B, K]
        k = torch.argmax(sims, dim=1)  # [B]
        # quantised table embedding for each chosen k -> conditions the VAE decode
        discrete_emb_table = self.action_rep.get_embedding(k)  # [B, dsc] (already tanh'd)

        # continuous: VAE decode over the batch -> x in (-max_action, max_action)
        x_cont, _delta = self.action_rep.vae.decode(st, true_parameter_emb, discrete_emb_table)
        if not explore and getattr(self, "track_stats", False):
            self.policy._record_stats(
                eval_latent_saturated=(parameter_emb.abs() > 0.99).float().mean(),
                eval_action_abs=x_cont.abs().mean(),
                eval_action_saturated=(x_cont.abs() > 0.99).float().mean())
        x = x_cont.detach().cpu().numpy().astype(np.float32).reshape(B, self.param_dim)
        k_np = k.detach().cpu().numpy().astype(np.int64)

        if explore:
            self._pending_batch = {
                "discrete_emb": discrete_emb.detach().cpu().numpy().astype(np.float32),
                "parameter_emb": parameter_emb.detach().cpu().numpy().astype(np.float32),
                "discrete_action": k_np,
                "parameter_action": x,
            }
        return k_np, x

    def store_batch_transition(self, state_bt, next_state_bt, reward_b, done_b):
        """Store B transitions for the latents produced by the most recent
        select_action_batch(explore=True). Adds to BOTH buffers, vectorized."""
        if getattr(self, "_pending_batch", None) is None:
            raise RuntimeError("store_batch_transition called before select_action_batch(explore=True).")
        p = self._pending_batch
        state_bt = np.asarray(state_bt, dtype=np.float32)
        next_state_bt = np.asarray(next_state_bt, dtype=np.float32)
        sns = next_state_bt - state_bt
        B = state_bt.shape[0]
        reward_b = np.asarray(reward_b, dtype=np.float32).reshape(B)
        done_b = np.broadcast_to(np.asarray(done_b, dtype=np.float32), (B,)).copy()
        zeros_d = np.zeros((B, self.reduced_action_dim), dtype=np.float32)
        zeros_p = np.zeros((B, self.parameter_emb_dim), dtype=np.float32)

        self.replay_buffer.add_batch(
            state_bt, p["discrete_action"], p["parameter_action"], p["parameter_action"],
            p["discrete_emb"], p["parameter_emb"], next_state_bt, sns, reward_b, done_b)
        self.replay_buffer_embedding.add_batch(
            state_bt, p["discrete_action"], p["parameter_action"], p["parameter_action"],
            zeros_d, zeros_p, next_state_bt, sns, reward_b, done_b)
        self._pending_batch = None

    def greedy_act_fn_batch(self):
        """Batched greedy act_fn for a vectorized eval loop: state[B,dim] -> (k[B], x[B,pdim])."""
        def act_fn(state_bt):
            return self.select_action_batch(state_bt, explore=False)
        return act_fn

    # ------------------------------------------------------------------ #
    #  Checkpoint / resume (weights + optimizers + running stats + LSC +   #
    #  replay buffers). Lets a run be EXTENDED later without restarting.   #
    # ------------------------------------------------------------------ #
    def save_state(self, ckpt_dir, save_buffers=True):
        """Save full learnable + running state to ckpt_dir. save_buffers=True also dumps the two
        replay buffers (compressed, filled-portion only) so off-policy training truly resumes."""
        import os as _os
        _os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            # networks + optimizers
            "vae": self.action_rep.vae.state_dict(),
            "vae_opt": self.action_rep.vae_optimizer.state_dict(),
            "actor": self.policy.actor.state_dict(),
            "actor_target": self.policy.actor_target.state_dict(),
            "actor_opt": self.policy.actor_optimizer.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "critic_target": self.policy.critic_target.state_dict(),
            "critic_opt": self.policy.critic_optimizer.state_dict(),
            # TD3 running state
            "total_it": self.policy.total_it,
            "rew_count": self.policy._rew_count, "rew_mean": self.policy._rew_mean,
            "rew_M2": self.policy._rew_M2, "normalize_reward": self.policy.normalize_reward,
            # LSC + representation state
            "c_rate": self.c_rate, "recon_s": self.recon_s, "delta_scale": self.delta_scale,
            "explore_calls": self._explore_calls,
            # dims (sanity-check on load)
            "dims": dict(state_dim=self.state_dim, K=self.K, param_dim=self.param_dim,
                         reduced_action_dim=self.reduced_action_dim,
                         parameter_emb_dim=self.parameter_emb_dim,
                         vae_hidden_size=self.vae_hidden_size),
        }, _os.path.join(ckpt_dir, "agent.pt"))
        if save_buffers:
            self.replay_buffer.save_npz(_os.path.join(ckpt_dir, "buffer_main.npz"))
            self.replay_buffer_embedding.save_npz(_os.path.join(ckpt_dir, "buffer_embed.npz"))

    def load_state(self, ckpt_dir, load_buffers=True):
        """Restore from save_state(). Rebuilds VAE at the saved delta_scale first so shapes match."""
        import os as _os
        ck = torch.load(_os.path.join(ckpt_dir, "agent.pt"), map_location=self.device)
        d = ck["dims"]
        assert (d["state_dim"], d["K"], d["param_dim"]) == (self.state_dim, self.K, self.param_dim), \
            f"checkpoint dims {d} != agent dims"
        # ensure VAE built with the saved delta_scale (delta head scale affects params' meaning)
        if abs(ck["delta_scale"] - self.delta_scale) > 1e-9:
            self._build_action_rep(ck["delta_scale"])
        self.action_rep.vae.load_state_dict(ck["vae"])
        self.action_rep.vae_optimizer.load_state_dict(ck["vae_opt"])
        self.policy.actor.load_state_dict(ck["actor"])
        self.policy.actor_target.load_state_dict(ck["actor_target"])
        self.policy.actor_optimizer.load_state_dict(ck["actor_opt"])
        self.policy.critic.load_state_dict(ck["critic"])
        self.policy.critic_target.load_state_dict(ck["critic_target"])
        self.policy.critic_optimizer.load_state_dict(ck["critic_opt"])
        self.policy.total_it = ck["total_it"]
        self.policy._rew_count = ck["rew_count"]; self.policy._rew_mean = ck["rew_mean"]
        self.policy._rew_M2 = ck["rew_M2"]; self.policy.normalize_reward = ck["normalize_reward"]
        self.c_rate = ck["c_rate"]; self.recon_s = ck["recon_s"]
        self.delta_scale = ck["delta_scale"]; self._explore_calls = ck["explore_calls"]
        if load_buffers:
            bm = _os.path.join(ckpt_dir, "buffer_main.npz")
            be = _os.path.join(ckpt_dir, "buffer_embed.npz")
            if _os.path.exists(bm): self.replay_buffer.load_npz(bm)
            if _os.path.exists(be): self.replay_buffer_embedding.load_npz(be)

    def store_last_transition(self, state, next_state, reward, done):
        """Store the transition for the latents produced by the most recent select_action().

        Adds to BOTH buffers (main -> TD3, embedding -> VAE), exactly as the reference loop.
        """
        if self._pending is None:
            raise RuntimeError("store_last_transition called before select_action.")
        p = self._pending
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        state_next_state = next_state - state

        self.replay_buffer.add(
            state, discrete_action=p["discrete_action"], parameter_action=p["parameter_action"],
            all_parameter_action=p["parameter_action"],
            discrete_emb=p["discrete_emb"], parameter_emb=p["parameter_emb"],
            next_state=next_state, state_next_state=state_next_state,
            reward=reward, done=done)
        self.replay_buffer_embedding.add(
            state, discrete_action=p["discrete_action"], parameter_action=p["parameter_action"],
            all_parameter_action=p["parameter_action"],
            discrete_emb=np.zeros(self.reduced_action_dim, dtype=np.float32),
            parameter_emb=np.zeros(self.parameter_emb_dim, dtype=np.float32),
            next_state=next_state, state_next_state=state_next_state,
            reward=reward, done=done)
        self._pending = None

    # ------------------------------------------------------------------ #
    #  One TD3 gradient update (latent space + inverting grads + RSC)     #
    # ------------------------------------------------------------------ #
    def train_step(self):
        """One TD3 update on the main buffer. No-op (returns None) until it has a full batch."""
        if self.replay_buffer.size < self.batch_size:
            return None
        if self.c_rate is None or self.recon_s is None:
            raise RuntimeError("LSC not initialised -- call warmup_and_pretrain_vae() first.")
        discrete_relable_rate, parameter_relable_rate = self.policy.train(
            self.replay_buffer, self.action_rep, self.c_rate, self.recon_s, self.batch_size)
        return discrete_relable_rate, parameter_relable_rate

    # convenience: greedy act_fn for env.eval_policy(act_fn)
    def greedy_act_fn(self):
        def act_fn(state):
            return self.select_action(state, explore=False)
        return act_fn
