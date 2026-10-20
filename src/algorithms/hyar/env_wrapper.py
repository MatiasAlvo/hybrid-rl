"""HyarPamdpWrapper -- boundary translator between HyAR (normalized latent world) and
the HPO-equivalent PamdpEnv (physical units).

WHY THIS EXISTS (fairness + representational adequacy)
------------------------------------------------------
HyAR's VAE decodes the continuous parameter through `max_action * tanh(...)`, so the agent
natively emits actions in (-1, 1)^param_dim, and its delta-state prediction head is likewise
`tanh(...)` -- it assumes ~unit-scale states and actions. The raw PamdpEnv, by contrast, is in
physical units:
  * S-LQR: control u is UNCONSTRAINED; analytic Riccati-optimal |u*| reaches ~4 (median) to ~21
    (tail) at small p (measured). A raw tanh cap of |u|<1 would make HyAR unable to represent the
    optimal control -- it would lose for a trivial representational reason, not an algorithmic one.
  * S-LQR states are O(10/sqrt(p)) with early blow-ups to +/-100; raw deltas would saturate the
    VAE's tanh delta-state head, degrading LSC/RSC.

This wrapper follows HyAR's OWN design (its reference envs use ScaledStateWrapper +
ScaledParameterisedActionWrapper): the agent stays entirely in normalized space (max_action=1),
and we translate ONLY at the env boundary:
    state (agent sees)  = (raw_state - state_mean) / state_scale
    u (sim sees)        = x_norm * action_scale          (x_norm in (-1,1))

FAIRNESS: the inner PamdpEnv is untouched and remains bit-for-bit HPO-equivalent (same
FeatureRegistry state, same dynamics, same loss/reported metric, same CRN scenarios). HPO itself
has an UNCONSTRAINED continuous head (disable_continuous_scaling for LQR), so giving HyAR a
generous action_scale does not advantage it -- it merely removes an artificial box HPO never had.
We VERIFY the box is non-binding by checking HyAR recovers the Riccati optimum at J=1 (the paper's
own Fig 8a ground-truth) before trusting any J>1 number.

The metric (`eval_policy` -> loss/reported) is computed by the INNER env in physical units, so
numbers are directly comparable to HPO regardless of the normalization applied to the agent.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch


class HyarPamdpWrapper:
    """Wrap a PamdpEnv so a HyAR agent (normalized (-1,1) actions, unit-scale states) can drive it.

    Args:
        inner:        a bridge.pamdp_env.PamdpEnv instance.
        action_scale: physical scale for the continuous action; agent x_norm in (-1,1) maps to
                      u = x_norm * action_scale. None -> derived default (see _derive_lqr_scales).
        state_scale:  divisor applied to raw states before the agent sees them. None -> derived.
        state_mean:   subtracted from raw states before scaling (default 0; LQR states center at 0).
        clip_state:   if set, clip normalized states to +/- clip_state (guards early LQR blow-ups
                      that the sim clamps to +/-100; keeps the VAE input bounded). Default 10.0.
    """

    def __init__(
        self,
        inner,
        action_scale: Optional[float] = None,
        state_scale: Optional[float] = None,
        state_mean: float = 0.0,
        clip_state: Optional[float] = 10.0,
        reward_scale: Optional[float] = None,
    ):
        self.inner = inner
        self.setting = inner.setting
        self.is_lqr = inner.is_lqr

        # ---- passthrough structural attributes (agent reads these) ----
        self.state_dim = inner.state_dim
        self.K = inner.K
        self.param_dim = inner.param_dim
        self.periods = inner.periods
        self.ignore_periods = inner.ignore_periods
        self.n_stores = inner.n_stores
        self.device = inner.device

        # ---- derive normalization scales ----
        d_state_scale, d_action_scale = self._derive_lqr_scales(inner)
        self.state_scale = float(state_scale) if state_scale is not None else d_state_scale
        self.action_scale = float(action_scale) if action_scale is not None else d_action_scale
        self.state_mean = float(state_mean)
        self.clip_state = clip_state

        if self.state_scale <= 0 or not math.isfinite(self.state_scale):
            self.state_scale = 1.0
        if self.action_scale <= 0 or not math.isfinite(self.action_scale):
            self.action_scale = 1.0

        # reward_scale: divide the reward HyAR LEARNS from by this (TD3 critic targets are
        # scale-sensitive; S-LQR costs are O(1000s) which destabilizes Q-learning). HPO also
        # standardizes costs by batch std (paper C.2), so scaling HyAR's reward is FAIR and
        # symmetric. The REPORTED metric (eval_policy* -> loss/reported) is computed by the inner
        # env in PHYSICAL units and is NEVER scaled -- only the training signal is. None -> derived
        # from warmup cost magnitudes at first use (set_reward_scale_from_costs), default 1.0.
        self.reward_scale = float(reward_scale) if reward_scale is not None else 1.0
        self._reward_scale_req = reward_scale

    # ------------------------------------------------------------------ #
    def _derive_lqr_scales(self, inner) -> Tuple[float, float]:
        """Scales from PUBLIC info (no privileged/infinite-horizon-optimal knowledge).

        state_scale  = r = init-state half-range / sqrt(p)  (states are driven toward 0; this keeps
                       the agent's state input ~O(1)).
        action_scale = 2 * p95(|u_1step|), where u_1step = -(R + B^T Q B)^-1 B^T Q A x is the ONE-STEP
                       greedy control (minimizes immediate stage cost). It uses ONLY the known model
                       (A,B,Q,R) — NO Riccati / infinite-horizon solve — so it is method-agnostic and
                       fair (any model-aware method could compute it; HPO's continuous head is
                       unbounded, i.e. implicitly has full range). Empirically 1-step p95 ~= true
                       Riccati |u*| p95 (verified), so 2x brackets it with headroom while keeping the
                       optimal action well inside HyAR's (-1,1) decode range (NOT crammed into ~20%).
        Previous default (4*r) was ~4x too large -> optimal action used only ~24% of range ->
        HANDICAPPED HyAR at high p. This fixes that.
        """
        if inner.is_lqr:
            try:
                ii = inner.cfg["store_params"]["initial_inventory"]
                lo, hi = ii["uniform_range"]
                half = max(abs(lo), abs(hi))
                p = inner.n_stores
                r = half / math.sqrt(p) if ii.get("scale_by_sqrt_dim", False) else half
                # 1-step greedy control p95 from the known model (no DARE)
                lqr = inner.problem_params.get("lqr", {})
                import numpy as _np
                A = _np.asarray(lqr["A"])[0]; B = _np.asarray(lqr["B"])[0]
                Q = _np.asarray(lqr["Q"])[0]; R = _np.asarray(lqr["R"])[0]
                M = _np.linalg.solve(R + B.T @ Q @ B, B.T @ Q @ A)   # u = -M x
                rng = _np.random.default_rng(0)
                Xs = rng.uniform(-r, r, size=(4000, p))
                u1 = -(M @ Xs.T).T
                a_scale = 2.0 * float(_np.percentile(_np.abs(u1), 95))
                a_scale = max(a_scale, 1e-3)
                return float(r), a_scale
            except Exception:
                # fallback: modest multiple of state scale (still far below the old 4*r bug intent)
                return float(r) if "r" in dir() else 1.0, float(2.0)
        # JRP: order qty >= 0 via softplus(x_norm * action_scale), x_norm in (-1,1). The needed
        # order over the lead time L is ~ L * per-store mean demand (plus safety stock), and demand
        # p99 can be ~2x the mean. softplus(action_scale) must comfortably exceed that, else HyAR
        # CANNOT order enough (structural loss -- the JRP analog of the LQR tanh cap). Derive
        # action_scale from the scenario demand: set it so softplus(action_scale) >= ~L*p99_demand.
        # State normalization: JRP obs are already mean-demand-normalized by the FeatureRegistry
        # (state ~O(1)), so state_scale=1 is correct (unlike LQR).
        try:
            L = int(inner.problem_params.get("lead_time", 2)) or 2
        except Exception:
            L = 2
        try:
            dem = inner._train_data["demands"].float()
            mean_dem = float(dem.mean().item())
            p99_dem = float(np.percentile(dem.cpu().numpy(), 99))
        except Exception:
            mean_dem, p99_dem = 10.0, 20.0
        # target max order ~ L * p99 demand + margin; invert softplus (~identity for large x) -> scale
        target_max_order = max(1.0, L * p99_dem)
        # softplus(a) ~= a for a>>0, so action_scale ~= target_max_order gives softplus(scale)>=target.
        action_scale = float(round(target_max_order, 1))
        return 1.0, action_scale

    # ------------------------------------------------------------------ #
    #  normalization helpers
    # ------------------------------------------------------------------ #
    def _norm_state_np(self, s: np.ndarray) -> np.ndarray:
        z = (np.asarray(s, dtype=np.float32) - self.state_mean) / self.state_scale
        if self.clip_state is not None:
            z = np.clip(z, -self.clip_state, self.clip_state)
        return z.astype(np.float32)

    def _norm_state_torch(self, s: torch.Tensor) -> torch.Tensor:
        z = (s - self.state_mean) / self.state_scale
        if self.clip_state is not None:
            z = z.clamp(-self.clip_state, self.clip_state)
        return z

    def _denorm_action_np(self, x_norm: np.ndarray) -> np.ndarray:
        return (np.asarray(x_norm, dtype=np.float32) * self.action_scale).astype(np.float32)

    def _denorm_action_torch(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.action_scale

    # ------------------------------------------------------------------ #
    #  single-scenario (online) API -- agent sees normalized world
    # ------------------------------------------------------------------ #
    def reset(self, scenario_idx: Optional[int] = None) -> np.ndarray:
        raw = self.inner.reset(scenario_idx)
        return self._norm_state_np(raw)

    def set_reward_scale_from_costs(self, costs):
        """Derive reward_scale from warmup per-step costs so scaled rewards are O(1). Uses the
        MEDIAN (robust) not RMS/mean: on an unstable system (A=1.1 I) random-policy costs have a
        huge blow-up tail (states diverge -> quadratic cost O(1e4+)); mean/RMS would over-scale
        massively (kills learning). Median reflects the typical operating cost. Mirrors HPO's
        cost standardization in spirit (HPO uses running batch std). No-op if pinned explicitly."""
        if self._reward_scale_req is not None:
            return self.reward_scale
        c = np.abs(np.asarray(costs, dtype=np.float64))
        med = float(np.median(c)) if c.size else 1.0
        self.reward_scale = max(1.0, round(med, 3))
        return self.reward_scale

    def step(self, k, x_norm):
        u = self._denorm_action_np(x_norm)              # (-1,1) -> physical
        raw_next, reward, done, info = self.inner.step(k, u)
        return self._norm_state_np(raw_next), reward / self.reward_scale, done, info

    # ------------------------------------------------------------------ #
    #  batched greedy evaluation -- metric computed by INNER env (physical)
    # ------------------------------------------------------------------ #
    def eval_policy(self, agent_act_fn: Callable, split: Optional[str] = None) -> Dict:
        """Run the agent greedily on the held-out set via the inner env's batched metric.

        `agent_act_fn` expects a NORMALIZED state and returns (k, x_norm) with x_norm in (-1,1).
        The inner env's eval_policy passes RAW states and treats the returned action as PHYSICAL,
        so this shim normalizes the state going in and denormalizes the action coming out.

        The inner act_fn call is batched ([B, state_dim]); HyARAgent.select_action is per-sample,
        so we loop over the batch here. (Eval is periodic; this is correctness-first. Vectorize if
        it becomes a wall-clock bottleneck on the full sweep.)
        """
        def shim(raw_state_batch: torch.Tensor):
            z = self._norm_state_torch(raw_state_batch)                 # [B, state_dim] normalized
            z_np = z.detach().cpu().numpy().astype(np.float32)
            B = z_np.shape[0]
            ks = np.empty(B, dtype=np.int64)
            xs = np.empty((B, self.param_dim), dtype=np.float32)
            for i in range(B):
                k_i, x_i = agent_act_fn(z_np[i])                        # (k, x_norm in (-1,1))
                ks[i] = int(k_i)
                xs[i] = np.asarray(x_i, dtype=np.float32).reshape(self.param_dim)
            u = xs * self.action_scale                                  # -> physical
            return ks, u

        return self.inner.eval_policy(shim, split=split)

    # ------------------------------------------------------------------ #
    #  BATCHED training rollout over the WHOLE training-scenario set.      #
    #  Drives the inner simulator over all B scenarios at once (the sim is #
    #  natively batched), reusing the inner env's VERIFIED action_dict     #
    #  recipe and state extractor so dynamics/state/metric are identical   #
    #  to the single-scenario path and to HPO. Agent sees NORMALIZED state.#
    # ------------------------------------------------------------------ #
    def _train_batch_reset(self):
        """Reset the inner online simulator on ALL training scenarios. Returns raw state [B,dim]."""
        inner = self.inner
        data = {k: v.clone() for k, v in inner._train_data.items()}
        obs, _ = inner._sim.reset(inner.periods, inner.problem_params, data, inner.observation_params)
        self._b_obs = obs
        self._b_t = 0
        raw = inner._extract_state(obs, inner._sim, inner._fr_online)  # [B, state_dim] torch
        self._b_raw = raw
        return raw

    def _train_batch_step(self, k_b, x_norm_b):
        """Step ALL training scenarios by one period with per-scenario (k,x_norm).

        k_b: int array [B]; x_norm_b: float array [B,param_dim] in (-1,1).
        Returns (raw_next[B,dim] torch, reward[B] np, done bool, cost[B] np).
        """
        inner = self.inner
        B = x_norm_b.shape[0]
        k_t = torch.as_tensor(k_b, device=inner.device).view(B)
        u = torch.as_tensor(x_norm_b, dtype=torch.float32, device=inner.device) * self.action_scale
        action_dict = inner._build_action_dict(k_t, u)
        next_obs, cost, _term, _, _ = inner._sim.step(self._b_obs, action_dict)
        self._b_obs = next_obs
        self._b_t += 1
        done = self._b_t >= inner.periods
        raw_next = inner._extract_state(next_obs, inner._sim, inner._fr_online)
        self._b_raw = raw_next
        cost_np = cost.detach().cpu().numpy().astype(np.float32).reshape(B)
        return raw_next, -cost_np / self.reward_scale, done, cost_np

    def train_rollout(self, agent, n_transitions, train_step_fn, utd=1, eval_every=None,
                      eval_cb=None, refresh_every_eps=10, verbose=True, vae_refresh_steps=1):
        """Run a scenario-BATCHED HyAR training loop until ~n_transitions env transitions.

        One "batched step" advances B scenarios by one period => B transitions counted.
        Per batched step we do `utd * B` ... no: we do TD3 updates at rate `utd` PER TRANSITION,
        i.e. round(utd * B) gradient updates per batched step (default utd=1 => B updates/step,
        matching the single-env "~1 grad step per env step" schedule). Episodes are `periods`
        long (the inner env horizon); every `refresh_every_eps` batched episodes the VAE takes
        `vae_refresh_steps` gradient steps (upstream HyAR: 1 step per 10 single episodes).

        agent must implement select_action_batch(state, explore) + store_batch_transition(...).
        train_step_fn is agent.train_step (one TD3 update; returns relable rates or None).
        eval_cb(transitions_done) is called every ~eval_every transitions (for logging).
        Returns the list of whatever eval_cb returned.
        """
        inner = self.inner
        B = inner.n_train
        updates_per_step = max(1, int(round(utd * B)))
        transitions = 0
        ep = 0
        results = []
        next_eval = 0

        while transitions < n_transitions:
            raw = self._train_batch_reset()
            z = self._norm_state_torch(raw).detach().cpu().numpy().astype(np.float32)
            for _t in range(inner.periods):
                k_b, x_b = agent.select_action_batch(z, explore=True)
                raw_next, reward_b, done, cost_b = self._train_batch_step(k_b, x_b)
                z_next = self._norm_state_torch(raw_next).detach().cpu().numpy().astype(np.float32)
                agent.store_batch_transition(z, z_next, reward_b, done)
                z = z_next
                transitions += B

                for _u in range(updates_per_step):
                    train_step_fn()

                if eval_every is not None and transitions >= next_eval:
                    if eval_cb is not None:
                        results.append(eval_cb(transitions))
                    next_eval += eval_every
                if transitions >= n_transitions:
                    break
            ep += 1
            if refresh_every_eps > 0 and ep % refresh_every_eps == 0:
                agent.refresh_vae(train_step=vae_refresh_steps, batch_size=64)

        return results

    @torch.no_grad()
    def eval_policy_batched(self, agent, split: Optional[str] = None) -> Dict:
        """Vectorized greedy eval: run agent.select_action_batch over the whole eval set, one
        period at a time, computing loss/reported identically to the inner env's eval_policy.

        Faster than eval_policy() (no per-sample python loop). Agent sees NORMALIZED state;
        action denormalized to physical before the inner simulator. Metric is in physical units.
        """
        inner = self.inner
        split = split or inner.eval_split
        from src.envs.pamdp.scenarios import split_runtime_params, make_scenarios
        periods, ignore_periods, _ = split_runtime_params(inner.cfg, split)
        if split == inner.eval_split:
            data = inner._eval_data
        else:
            cache = self.__dict__.setdefault("_split_data", {})
            if split not in cache:
                cache[split] = inner._to_device(make_scenarios(inner.setting, split, cfg=inner.cfg))
            data = cache[split]
        data = {k: v.clone() for k, v in data.items()}
        B = int(data["initial_inventories"].shape[0])

        obs, _ = inner._eval_sim.reset(periods, inner.problem_params, data, inner.observation_params)
        sum_total = 0.0
        sum_reported = 0.0
        for t in range(periods):
            raw = inner._extract_state(obs, inner._eval_sim, inner._fr_eval)  # [B,dim]
            z = self._norm_state_torch(raw).detach().cpu().numpy().astype(np.float32)
            k_b, x_b = agent.select_action_batch(z, explore=False)
            k_t = torch.as_tensor(k_b, device=inner.device).view(B)
            u = torch.as_tensor(x_b, dtype=torch.float32, device=inner.device) * self.action_scale
            action_dict = inner._build_action_dict(k_t, u)
            obs, cost, _term, _, _ = inner._eval_sim.step(obs, action_dict)
            bc = float(cost.sum().item())
            sum_total += bc
            if t >= ignore_periods:
                sum_reported += bc
        periods_tracking = periods - ignore_periods
        return {
            "loss/reported": sum_reported / (B * periods_tracking * inner.n_stores),
            "loss/total": sum_total / (B * periods * inner.n_stores),
            "n_samples": B, "periods": periods, "ignore_periods": ignore_periods,
            "n_stores": inner.n_stores,
        }

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"HyarPamdpWrapper(setting={self.setting!r}, p={self.n_stores}, K={self.K}, "
            f"param_dim={self.param_dim}, state_dim={self.state_dim}, "
            f"state_scale={self.state_scale:.4g}, action_scale={self.action_scale:.4g}, "
            f"clip_state={self.clip_state})"
        )
