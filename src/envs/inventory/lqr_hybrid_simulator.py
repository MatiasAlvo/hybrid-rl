from src import torch
from src.envs.inventory.hybrid_simulator import HybridSimulator


class LqrHybridSimulator(HybridSimulator):
    """
    Hybrid simulator for switched LQR dynamics with a global discrete mode
    and per-store continuous controls.
    """
    def reset(self, periods, problem_params, data, observation_params):
        observation, info = super().reset(periods, problem_params, data, observation_params)
        self._load_lqr_params(problem_params)
        return observation, info

    def _load_lqr_params(self, problem_params):
        lqr_params = problem_params.get('lqr', {})
        missing = [key for key in ('A', 'B', 'Q', 'R') if key not in lqr_params]
        if missing:
            raise ValueError(f"Missing LQR parameters: {missing}. Expected keys: A, B, Q, R.")

        self.lqr_A = self._to_tensor(lqr_params['A'])
        self.lqr_B = self._to_tensor(lqr_params['B'])
        self.lqr_Q = self._to_tensor(lqr_params['Q'])
        self.lqr_R = self._to_tensor(lqr_params['R'])

        if self.lqr_A.dim() != 3 or self.lqr_Q.dim() != 3:
            raise ValueError("LQR A and Q must have shape [n_modes, n_state, n_state].")
        if self.lqr_B.dim() != 3 or self.lqr_R.dim() != 3:
            raise ValueError("LQR B and R must have shape [n_modes, n_state, n_action] and [n_modes, n_action, n_action].")

        n_modes, n_state, _ = self.lqr_A.shape
        if self.lqr_B.shape[0] != n_modes or self.lqr_Q.shape[0] != n_modes or self.lqr_R.shape[0] != n_modes:
            raise ValueError("LQR A, B, Q, R must have the same number of modes.")

        n_action = self.lqr_B.shape[2]
        if self.lqr_Q.shape[1:] != (n_state, n_state):
            raise ValueError("LQR Q must match state dimension.")
        if self.lqr_R.shape[1:] != (n_action, n_action):
            raise ValueError("LQR R must match action dimension.")

        if n_state != self.n_stores:
            raise ValueError(
                f"LQR state dimension ({n_state}) must match n_stores ({self.n_stores}) "
                "for the inventory mapping."
            )

        self.lqr_n_modes = n_modes
        self.lqr_state_dim = n_state
        self.lqr_action_dim = n_action

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def step(self, observation, action_dict):
        current_observation = self._internal_observation
        next_observation, costs = self._calculate_lqr_transition_and_costs(current_observation, action_dict)

        self._internal_observation = next_observation.copy()

        if self.normalize_observations:
            self._inventory_observations.append(next_observation['store_inventories'][..., 0].detach())
            next_observation = self._normalize_observation(next_observation)

        return next_observation, costs, False, {}, {}

    def _calculate_lqr_transition_and_costs(self, observation, action_dict):
        x = self._extract_state_vector(observation)
        u = self._extract_action_vector(action_dict)
        mode_idx = self._extract_mode_indices(action_dict)

        A = self.lqr_A[mode_idx]
        B = self.lqr_B[mode_idx]
        Q = self.lqr_Q[mode_idx]
        R = self.lqr_R[mode_idx]

        x_next = torch.einsum('bij,bj->bi', A, x) + torch.einsum('bij,bj->bi', B, u)
        x_next = torch.clamp(x_next, max=100.0, min=-100.0)

        state_cost = torch.einsum('bi,bij,bj->b', x, Q, x)
        action_cost = torch.einsum('bi,bij,bj->b', u, R, u)
        total_cost = state_cost + action_cost

        next_observation = observation.copy()
        next_observation['store_inventories'] = self._embed_state_vector(x_next, observation['store_inventories'])
        next_observation['current_period'] += 1

        return next_observation, total_cost

    def _extract_state_vector(self, observation):
        inventory = observation['store_inventories']
        if inventory.dim() == 3:
            x = inventory[:, :, 0]
        elif inventory.dim() == 2:
            x = inventory
        else:
            raise ValueError("store_inventories must have shape [B, S] or [B, S, L].")

        if x.shape[1] != self.lqr_state_dim:
            raise ValueError(
                f"State dimension mismatch: expected {self.lqr_state_dim}, got {x.shape[1]}."
            )
        return x

    def _extract_action_vector(self, action_dict):
        feature_actions = action_dict.get('feature_actions')
        total_action = None if feature_actions is None else feature_actions.get('total_action')
        if total_action is None:
            raise ValueError(
                "LQR simulator requires action_dict['feature_actions']['total_action'] to match HybridSimulator."
            )
        if total_action.dim() == 2:
            u = total_action
        else:
            raise ValueError("feature_actions['total_action'] must have shape [B, A].")

        if u.shape[1] < self.lqr_action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected at least {self.lqr_action_dim}, got {u.shape[1]}."
            )
        if u.shape[1] > self.lqr_action_dim:
            u = u[:, :self.lqr_action_dim]
        return u

    def _extract_mode_indices(self, action_dict):
        if 'discrete_action_indices' not in action_dict or action_dict['discrete_action_indices'] is None:
            raise ValueError("LQR simulator requires action_dict['discrete_action_indices'].")

        mode_idx = action_dict['discrete_action_indices']
        if mode_idx.dim() > 1:
            mode_idx = mode_idx[:, 0]
        mode_idx = mode_idx.to(self.device)

        if mode_idx.max().item() >= self.lqr_n_modes or mode_idx.min().item() < 0:
            raise ValueError(f"Discrete mode index out of range: {mode_idx.min().item()}..{mode_idx.max().item()}")

        return mode_idx.long()

    def _embed_state_vector(self, x_next, inventory_template):
        if inventory_template.dim() == 2:
            return x_next
        new_inventory = torch.zeros_like(inventory_template)
        new_inventory[:, :, 0] = x_next
        return new_inventory
