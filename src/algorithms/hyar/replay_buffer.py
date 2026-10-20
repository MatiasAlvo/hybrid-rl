# Adapted from HyAR (TJU-DRL-LAB/self-supervised-rl), Apache-2.0; see LICENSE and NOTICE.md here.
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim,all_parameter_action_dim ,discrete_emb_dim, parameter_emb_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, discrete_action, parameter_action, all_parameter_action,discrete_emb, parameter_emb, next_state,state_next_state, reward, done):
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, state, discrete_action, parameter_action, all_parameter_action,
                  discrete_emb, parameter_emb, next_state, state_next_state, reward, done):
        """Vectorized add of B transitions at once (handles ring-buffer wraparound).

        All args are arrays with leading dim B. discrete_action/reward/done may be [B] or [B,1].
        Semantics are identical to calling add() B times in order.
        """
        B = state.shape[0]
        idx = (self.ptr + np.arange(B)) % self.max_size

        def _col(a, width):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 1:
                a = a.reshape(B, width)
            return a

        self.state[idx] = state
        self.discrete_action[idx] = _col(discrete_action, self.discrete_action.shape[1])
        self.parameter_action[idx] = parameter_action
        self.all_parameter_action[idx] = all_parameter_action
        self.discrete_emb[idx] = discrete_emb
        self.parameter_emb[idx] = parameter_emb
        self.next_state[idx] = next_state
        self.state_next_state[idx] = state_next_state
        self.reward[idx] = _col(reward, 1)
        self.not_done[idx] = 1.0 - _col(done, 1)

        self.ptr = (self.ptr + B) % self.max_size
        self.size = min(self.size + B, self.max_size)

    def save_npz(self, path):
        """Save only the FILLED portion (size rows) as compressed float32 -> resumable, compact."""
        s = self.size
        np.savez_compressed(
            path, ptr=self.ptr, size=self.size, max_size=self.max_size,
            state=self.state[:s].astype(np.float32),
            discrete_action=self.discrete_action[:s].astype(np.float32),
            parameter_action=self.parameter_action[:s].astype(np.float32),
            all_parameter_action=self.all_parameter_action[:s].astype(np.float32),
            discrete_emb=self.discrete_emb[:s].astype(np.float32),
            parameter_emb=self.parameter_emb[:s].astype(np.float32),
            next_state=self.next_state[:s].astype(np.float32),
            state_next_state=self.state_next_state[:s].astype(np.float32),
            reward=self.reward[:s].astype(np.float32),
            not_done=self.not_done[:s].astype(np.float32))

    def load_npz(self, path):
        d = np.load(path)
        s = int(d["size"])
        self.ptr = int(d["ptr"]); self.size = s
        for k in ("state", "discrete_action", "parameter_action", "all_parameter_action",
                  "discrete_emb", "parameter_emb", "next_state", "state_next_state",
                  "reward", "not_done"):
            getattr(self, k)[:s] = d[k]

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.discrete_action[ind]).to(self.device),
            torch.FloatTensor(self.parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.all_parameter_action[ind]).to(self.device),
            torch.FloatTensor(self.discrete_emb[ind]).to(self.device),
            torch.FloatTensor(self.parameter_emb[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.state_next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class TorchReplayBuffer(object):
    """Replay buffer with the same interface as ReplayBuffer, stored as float32 tensors on one device.

    Keeping the buffer on the GPU avoids copying ten float64 arrays from host memory on every
    TD3 update, which dominates the update cost for small networks. Sampling uses torch's RNG,
    so runs are not bit-identical to ReplayBuffer runs.
    """

    _FIELDS = ("state", "discrete_action", "parameter_action", "all_parameter_action",
               "discrete_emb", "parameter_emb", "next_state", "state_next_state", "reward", "not_done")

    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, all_parameter_action_dim,
                 discrete_emb_dim, parameter_emb_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        widths = (state_dim, discrete_action_dim, parameter_action_dim, all_parameter_action_dim,
                  discrete_emb_dim, parameter_emb_dim, state_dim, state_dim, 1, 1)
        for name, width in zip(self._FIELDS, widths):
            setattr(self, name, torch.zeros((max_size, width), dtype=torch.float32, device=self.device))

    def _as_rows(self, value, rows, width):
        t = torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)
        return t.reshape(rows, width)

    def add(self, state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb,
            next_state, state_next_state, reward, done):
        self.add_batch(np.asarray(state)[None], [discrete_action], np.asarray(parameter_action)[None],
                       np.asarray(all_parameter_action)[None], np.asarray(discrete_emb)[None],
                       np.asarray(parameter_emb)[None], np.asarray(next_state)[None],
                       np.asarray(state_next_state)[None], [reward], [done])

    def add_batch(self, state, discrete_action, parameter_action, all_parameter_action,
                  discrete_emb, parameter_emb, next_state, state_next_state, reward, done):
        B = np.asarray(state).shape[0]
        idx = torch.as_tensor((self.ptr + np.arange(B)) % self.max_size, device=self.device)
        values = (state, discrete_action, parameter_action, all_parameter_action, discrete_emb,
                  parameter_emb, next_state, state_next_state, reward, 1.0 - np.asarray(done, dtype=np.float32))
        for name, value in zip(self._FIELDS, values):
            buf = getattr(self, name)
            buf[idx] = self._as_rows(value, B, buf.shape[1])
        self.ptr = (self.ptr + B) % self.max_size
        self.size = min(self.size + B, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return tuple(getattr(self, name)[ind] for name in self._FIELDS)
