"""A tiny self-contained MOCK PAMDP satisfying HyAR/bridge/CONTRACT.md.

This is ONLY for smoke-testing the HyAR port end to end without the real
`PamdpEnv` (which is being built by a parallel effort). It is deliberately
*learnable*: there is a state-dependent optimal discrete action k*(s) and a
state-dependent optimal continuous parameter x*(s), so a working HyAR loop
should drive `loss/reported` (mean cost) down from the random baseline.

Contract surface implemented:
  attributes: state_dim (int), K (int), param_dim (int)
  reset(scenario_idx=None) -> state
  step(k, x) -> (next_state, reward, done, info)   reward = -cost,
      info = {'cost': float, 'reported': bool}
  eval_policy(act_fn) -> {'loss/reported': float, ...}   greedy over test set

Cost model (per step):
  cost = c_disc * 1[k != k*(s)] + mean_j (x_j - x*(s)_j)^2
  k*(s)  = argmax(s[:K])            (state-dependent discrete optimum)
  x*(s)_j = x_scale * tanh(s[j % state_dim])   in (-x_scale, x_scale)
HyAR decodes x in (-1,1) (VAE tanh), so x*(s) with x_scale<1 is reachable.
"""

import numpy as np


class MockPamdpEnv(object):
    def __init__(self, state_dim=6, K=3, param_dim=4, periods=1, ignore_periods=0,
                 n_train=256, n_test=128, c_disc=1.0, x_scale=0.6, seed=0):
        self.state_dim = int(state_dim)
        self.K = int(K)
        self.param_dim = int(param_dim)
        self.periods = int(periods)
        self.ignore_periods = int(ignore_periods)
        self.c_disc = float(c_disc)
        self.x_scale = float(x_scale)

        rng = np.random.RandomState(seed)
        # Fixed scenario sets (CRN): each scenario is an initial state vector.
        self._train_states = rng.randn(n_train, self.state_dim).astype(np.float32)
        self._test_states = rng.randn(n_test, self.state_dim).astype(np.float32)
        self._eval_rng = np.random.RandomState(seed + 1)

        self._state = None
        self._t = 0

    # -------- optima / cost --------
    def _k_star(self, s):
        return int(np.argmax(s[:self.K]))

    def _x_star(self, s):
        idx = np.arange(self.param_dim) % self.state_dim
        return (self.x_scale * np.tanh(s[idx])).astype(np.float32)

    def _cost(self, s, k, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        disc = self.c_disc * (1.0 if int(k) != self._k_star(s) else 0.0)
        cont = float(np.mean((x - self._x_star(s)) ** 2))
        return disc + cont

    def _transition(self, s, x):
        if self.periods <= 1:
            return s  # ignored (episode ends)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        idx = np.arange(self.state_dim) % self.param_dim
        return (0.9 * s + 0.1 * x[idx]).astype(np.float32)

    # -------- contract API --------
    def reset(self, scenario_idx=None):
        if scenario_idx is None:
            i = np.random.randint(0, len(self._train_states))
            self._state = self._train_states[i].copy()
        else:
            self._state = self._train_states[int(scenario_idx) % len(self._train_states)].copy()
        self._t = 0
        return self._state.copy()

    def step(self, k, x):
        s = self._state
        cost = self._cost(s, k, x)
        reported = (self._t >= self.ignore_periods)
        self._t += 1
        done = self._t >= self.periods
        next_state = self._transition(s, x)
        self._state = next_state
        info = {"cost": float(cost), "reported": bool(reported)}
        return next_state.copy(), float(-cost), bool(done), info

    def eval_policy(self, act_fn):
        """Greedy rollout of act_fn over ALL test scenarios; mean per-period reported cost."""
        costs = []
        for i in range(len(self._test_states)):
            s = self._test_states[i].copy()
            for t in range(self.periods):
                k, x = act_fn(s)
                c = self._cost(s, k, x)
                if t >= self.ignore_periods:
                    costs.append(c)
                s = self._transition(s, x)
        mean_cost = float(np.mean(costs)) if costs else float("nan")
        return {"loss/reported": mean_cost, "n_eval_periods": len(costs)}

    # -------- reference baselines (for the smoke test only) --------
    def random_baseline(self):
        costs = []
        for s in self._test_states:
            k = np.random.randint(0, self.K)
            x = np.random.uniform(-1.0, 1.0, size=self.param_dim).astype(np.float32)
            costs.append(self._cost(s, k, x))
        return float(np.mean(costs))

    def optimal_baseline(self):
        costs = []
        for s in self._test_states:
            costs.append(self._cost(s, self._k_star(s), self._x_star(s)))
        return float(np.mean(costs))
