"""
pamdp_env.py -- PAMDP environment adapter wrapping the hybrid-rl JRP and switched-LQR
simulators so an external agent (HyAR) can drive them.

Both settings reduce to a parameterized-action MDP: at each step pick a discrete action
`k in {0..K-1}` and a continuous parameter vector `x in R^{param_dim}`.

  * JRP  (fixed_costs_main):  K = 2  (0=don't-order, 1=order).  param_dim = p = n_stores.
        Continuous param -> per-product order quantities via softplus (>=0), zeroed when
        k = 0. Fixed cost K_fix = 64*p charged iff k = 1.
  * S-LQR (lqr_switched_main): K = J = n_modes.  param_dim = p = n_stores.
        Continuous param = control u in R^p, UNCONSTRAINED (no activation). k = LQR mode.

FAIRNESS: the `state` handed to the agent is the SAME feature vector HPO's policy ingests,
produced by the repo FeatureRegistry.prepare_inputs (identical to trainer.simulate_batch,
which feeds model(obs) -> feature_registry.prepare_inputs(obs)). For LQR that is raw x in
R^p; for JRP it is HPO's processed obs (store_inventories + costs + mean, normalized by mean
demand, plus moving_mean and K/mean features), flattened.

Interface:
  * reset(scenario_idx=None) -> state           single-scenario episode (gym-style, np array)
  * step(k, x) -> (next_state, reward, done, info)   reward = -cost; done after `periods`
  * eval_policy(act_fn) -> dict                 batched greedy rollout over the eval set,
        returns 'loss/reported' identical to trainer.py:375.

Setting dicts are deep-copied and overridden in memory only; no YAML on disk is modified.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.inventory.range_manager import RangeManager
from src.features.feature_registry import FeatureRegistry
from src.envs.inventory.hybrid_simulator import HybridSimulator
from src.envs.inventory.lqr_hybrid_simulator import LqrHybridSimulator

from src.envs.pamdp.scenarios import (
    prepare_setting_config,
    load_policy_config,
    make_scenarios,
    split_runtime_params,
    SETTING_REGISTRY,
)

ArrayLike = Union[np.ndarray, torch.Tensor, list, float, int]


class PamdpEnv:
    """PAMDP adapter over one hybrid-rl simulator (JRP or switched-LQR)."""

    def __init__(
        self,
        setting: str,
        n_stores: Optional[int] = None,
        n_modes: Optional[int] = None,
        policy_stem: Optional[str] = None,
        eval_split: str = "test",
        device: Optional[str] = None,
        sample_seed: int = 0,
    ):
        """
        Args:
            setting     : 'jrp' | 'lqr'.
            n_stores    : override p (in-memory; YAML untouched). None -> config default.
            n_modes     : override J for LQR (in-memory). None -> config default.
            policy_stem : HPO policy YAML stem defining the FeatureRegistry / state. None
                          -> the canonical HPO policy for the setting (see SETTING_REGISTRY).
            eval_split  : which split eval_policy scores on ('test' by default -- held-out).
            device      : torch device string. None -> 'cuda:0' if available else 'cpu'.
            sample_seed : RNG seed for reset(scenario_idx=None) uniform sampling.
        """
        if setting not in ("jrp", "lqr"):
            raise ValueError(f"setting must be 'jrp' or 'lqr', got {setting!r}")
        self.setting = setting
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---- prepared, in-memory setting config (scaled JRP cost / injected LQR mats) ----
        self.cfg = prepare_setting_config(setting, n_stores=n_stores, n_modes=n_modes)
        self.problem_params = self.cfg["problem_params"]
        self.observation_params = self.cfg["observation_params"]
        self.n_stores = int(self.problem_params["n_stores"])
        self.n_modes = int(self.problem_params.get("n_modes", 0)) or None

        # ---- policy config -> RangeManager + FeatureRegistry (exactly as main_run) --------
        self.policy_config = load_policy_config(setting, policy_stem)
        self.nn_params = self.policy_config["nn_params"]
        rm_cfg = self.problem_params.copy()
        rm_cfg.update(self.nn_params["policy_network"])
        self.range_manager = RangeManager(rm_cfg, device=self.device)
        net_dims = self.range_manager.get_network_dimensions()
        # main_run.py:306-311 writes the head sizes back before building FeatureRegistry.
        self.nn_params["policy_network"]["heads"]["discrete"]["size"] = net_dims["n_discrete"]
        self.nn_params["policy_network"]["heads"]["continuous"]["size"] = net_dims["n_continuous"]
        # Separate FeatureRegistry instances so online (reset/step) and batched eval rollouts
        # never share EWMA state (feature_registry.prepare_inputs is stateful across a rollout).
        self._fr_online = FeatureRegistry(self.nn_params, self.range_manager)
        self._fr_eval = FeatureRegistry(self.nn_params, self.range_manager)

        # ---- discrete / continuous structure --------------------------------------------
        self.is_lqr = self.problem_params.get("simulator_type") == "lqr_hybrid"
        self.param_dim = self.n_stores
        if self.is_lqr:
            self.K = self.n_modes
        else:
            self.K = 2  # JRP: don't-order / order (single global binary)
            fc = self.problem_params["discrete_features"]["fixed_ordering_cost"]
            self.K_fix = float(fc["values"][1])  # == 64 * n_stores after scaling
            self._fixed_cost_values = torch.tensor(
                [float(fc["values"][0]), self.K_fix], device=self.device, dtype=torch.float32
            )

        # ---- runtime periods for online episodes (train split) ---------------------------
        self._train_periods, self._train_ignore, _ = split_runtime_params(self.cfg, "train")
        self.periods = self._train_periods
        self.ignore_periods = self._train_ignore

        # ---- scenario data (train for online rollouts; eval_split for scoring) -----------
        self._train_data = self._to_device(make_scenarios(setting, "train", cfg=self.cfg))
        self.eval_split = eval_split
        self._eval_data = self._to_device(make_scenarios(setting, eval_split, cfg=self.cfg))
        self.n_train = int(self._train_data["initial_inventories"].shape[0])
        self.n_eval = int(self._eval_data["initial_inventories"].shape[0])
        self._rng = np.random.default_rng(sample_seed)

        # ---- simulators (one for online, one for eval) -----------------------------------
        sim_cls = LqrHybridSimulator if self.is_lqr else HybridSimulator
        self._sim = sim_cls(self._fr_online, model=None, device=self.device)
        self._eval_sim = sim_cls(self._fr_eval, model=None, device=self.device)

        # ---- state_dim: measured via a throwaway 2-sample eval reset ---------------------
        self.state_dim = self._measure_state_dim()

        # online-episode bookkeeping
        self._obs = None
        self._t = 0
        self._episode_active = False

    # ===================================================================================== #
    # helpers
    # ===================================================================================== #
    def _to_device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in data.items()}

    def _slice_data(self, data: Dict[str, torch.Tensor], idx) -> Dict[str, torch.Tensor]:
        """Select a subset of scenarios (list/array of indices) as a fresh data dict."""
        idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        return {k: v.index_select(0, idx_t).clone() for k, v in data.items()}

    def _extract_state(self, obs: Dict, sim, feature_registry) -> torch.Tensor:
        """
        Reproduce trainer.simulate_batch's state exactly: add internal_data (as the trainer's
        _prepare_observation_with_internal_data does, trainer.py:584-588) then run
        feature_registry.prepare_inputs (what model.forward calls, hybrid_agent.py:291).
        update_ewma=True so the moving-demand normalization evolves identically to HPO.
        """
        obs_and_internal = {k: v for k, v in obs.items()}
        obs_and_internal["internal_data"] = sim._internal_data
        return feature_registry.prepare_inputs(obs_and_internal, update_ewma=True)

    def _measure_state_dim(self) -> int:
        n = min(2, self.n_eval)
        data = self._slice_data(self._eval_data, list(range(n)))
        periods, _, _ = split_runtime_params(self.cfg, self.eval_split)
        obs, _ = self._eval_sim.reset(periods, self.problem_params, data, self.observation_params)
        vec = self._extract_state(obs, self._eval_sim, self._fr_eval)
        return int(vec.shape[-1])

    def _build_action_dict(self, k: torch.Tensor, x: torch.Tensor) -> Dict:
        """
        Build the simulator action_dict from batched (k[B], x[B, param_dim]) using the
        validated recipes (checked against the trainer by tests/hyar/test_pamdp_env.py).
        """
        B = x.shape[0]
        if self.is_lqr:
            # S-LQR: control unconstrained; discrete = mode index.
            u = x.to(device=self.device, dtype=torch.float32)
            mode = k.to(device=self.device, dtype=torch.long).view(B)
            return {
                "feature_actions": {"total_action": u},
                "discrete_action_indices": mode,
            }
        # JRP: order qty >= 0 via softplus; zeroed when not ordering. Global binary fixed cost.
        do_order = k.to(device=self.device, dtype=torch.float32).view(B)
        qty = F.softplus(x.to(device=self.device, dtype=torch.float32))  # [B, p] >= 0
        qty = qty * do_order.unsqueeze(1)  # zero rows where k == 0
        range_probs = torch.zeros(B, 1, 2, device=self.device, dtype=torch.float32)
        range_probs[:, 0, 0] = 1.0 - do_order
        range_probs[:, 0, 1] = do_order
        return {
            "feature_actions": {
                "total_action": qty,
                "fixed_ordering_cost": {
                    "range_probs": range_probs,
                    "values": self._fixed_cost_values,
                },
            }
        }

    @staticmethod
    def _as_batched_k(k: ArrayLike, B: int) -> torch.Tensor:
        t = torch.as_tensor(k)
        return t.view(-1) if t.dim() > 0 else t.view(1).expand(B)

    def _as_batched_x(self, x: ArrayLike, B: int) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[0] != B:
            t = t.expand(B, -1)
        return t

    # ===================================================================================== #
    # single-scenario (online) API
    # ===================================================================================== #
    def reset(self, scenario_idx: Optional[int] = None) -> np.ndarray:
        """
        Start an episode on ONE training scenario (one demand trace / one s0). If
        scenario_idx is None, sample uniformly from the training set. Deterministic given
        idx (CRN: identical exogenous trace -> identical trajectory under a fixed policy).

        Returns the initial state as a float32 np.ndarray of shape [state_dim].
        """
        if scenario_idx is None:
            scenario_idx = int(self._rng.integers(self.n_train))
        self._scenario_idx = int(scenario_idx)

        data = self._slice_data(self._train_data, [self._scenario_idx])
        obs, _ = self._sim.reset(self.periods, self.problem_params, data, self.observation_params)
        self._obs = obs
        self._t = 0
        self._episode_active = True
        state = self._extract_state(obs, self._sim, self._fr_online)
        return state.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def step(self, k: Union[int, ArrayLike], x: ArrayLike):
        """
        Apply parameterized action (k, x) for one period on the current online episode.

        Args:
            k : discrete action, int in {0..K-1}.
            x : continuous parameter, shape [param_dim] (np array or torch tensor).
        Returns:
            (next_state[np, state_dim], reward: float, done: bool, info: dict)
            reward = -cost; done = True after `periods` steps.
            info = {'cost': float (per-period cost summed over stores),
                    'reported': bool (t >= ignore_periods),
                    't': int, 'scenario_idx': int}
        """
        if not self._episode_active:
            raise RuntimeError("step() called before reset() (or after episode done).")

        k_t = self._as_batched_k(k, 1)
        x_t = self._as_batched_x(x, 1)
        action_dict = self._build_action_dict(k_t, x_t)

        next_obs, cost, _terminated, _, _ = self._sim.step(self._obs, action_dict)
        self._obs = next_obs

        cost_val = float(cost.reshape(-1)[0].item())
        reported = self._t >= self.ignore_periods
        info = {
            "cost": cost_val,
            "reported": reported,
            "t": self._t,
            "scenario_idx": self._scenario_idx,
        }

        self._t += 1
        done = self._t >= self.periods
        if done:
            self._episode_active = False

        next_state = self._extract_state(next_obs, self._sim, self._fr_online)
        next_state = next_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return next_state, -cost_val, done, info

    # ===================================================================================== #
    # batched greedy evaluation
    # ===================================================================================== #
    def eval_policy(
        self,
        act_fn: Callable[[torch.Tensor], Tuple[ArrayLike, ArrayLike]],
        split: Optional[str] = None,
    ) -> Dict:
        """
        Run act_fn greedily over ALL scenarios in the eval set in ONE batched rollout and
        return the paper's headline scalar 'loss/reported'.

        act_fn signature (batched):
            act_fn(state) -> (k, x)
            state : torch.FloatTensor [B, state_dim] on self.device.
            k     : array-like [B] of discrete indices in {0..K-1}.
            x     : array-like [B, param_dim] of continuous params.

        Metric fidelity -- reproduces trainer.py exactly:
            per step, sim.step returns cost[B] (per-sample, already summed over stores;
            simulator.py:230 / hybrid_simulator.py:135). The trainer accumulates
            reward_to_report = sum over t>=ignore of cost.sum() over the batch
            (trainer.py:476,500-502) and reports
            loss/reported = epoch_loss_to_report / (n_samples*(periods-ignore)*n_stores)
            (trainer.py:375). loss/total uses all periods (trainer.py:374). We do the
            identical accumulation and normalization over the whole eval set in one batch.

        Returns dict with 'loss/reported', 'loss/total', and the raw sums / shapes used
        (so the normalization can be checked independently).
        """
        split = split or self.eval_split
        periods, ignore_periods, _ = split_runtime_params(self.cfg, split)
        data = self._eval_data if split == self.eval_split else self._to_device(
            make_scenarios(self.setting, split, cfg=self.cfg)
        )
        B = int(data["initial_inventories"].shape[0])
        data = {k: v.clone() for k, v in data.items()}

        obs, _ = self._eval_sim.reset(periods, self.problem_params, data, self.observation_params)

        sum_total = 0.0
        sum_reported = 0.0
        with torch.no_grad():
            for t in range(periods):
                state = self._extract_state(obs, self._eval_sim, self._fr_eval)
                k_raw, x_raw = act_fn(state)
                k_t = self._as_batched_k(k_raw, B).to(self.device)
                x_t = self._as_batched_x(x_raw, B).to(self.device)
                action_dict = self._build_action_dict(k_t, x_t)
                obs, cost, _term, _, _ = self._eval_sim.step(obs, action_dict)
                batch_cost = float(cost.sum().item())
                sum_total += batch_cost
                if t >= ignore_periods:
                    sum_reported += batch_cost

        periods_tracking = periods - ignore_periods
        denom_total = B * periods * self.n_stores
        denom_reported = B * periods_tracking * self.n_stores
        return {
            "loss/reported": sum_reported / denom_reported,
            "loss/total": sum_total / denom_total,
            "sum_reported": sum_reported,
            "sum_total": sum_total,
            "n_samples": B,
            "periods": periods,
            "ignore_periods": ignore_periods,
            "periods_tracking": periods_tracking,
            "n_stores": self.n_stores,
        }

    # ===================================================================================== #
    # convenience
    # ===================================================================================== #
    def __repr__(self) -> str:
        return (
            f"PamdpEnv(setting={self.setting!r}, p={self.n_stores}, "
            f"K={self.K}, param_dim={self.param_dim}, state_dim={self.state_dim}, "
            f"periods={self.periods}, ignore={self.ignore_periods}, device={self.device!r})"
        )
