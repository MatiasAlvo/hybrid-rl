"""Checks that the opt-in speed paths compute the same updates as the upstream code paths."""
import copy

import numpy as np
import torch

from src.algorithms.hyar import action_representation, td3
from src.algorithms.hyar.replay_buffer import ReplayBuffer, TorchReplayBuffer


def _setup(state_dim=5, K=3, param_dim=4, latent=6, emb=4, n=512, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rep = action_representation.Action_representation(
        state_dim=state_dim, action_dim=K, parameter_action_dim=param_dim,
        reduced_action_dim=emb, reduce_parameter_action_dim=latent)
    policy = td3.TD3(state_dim, emb, latent, max_action=1.0, policy_noise=0.1)
    buf = ReplayBuffer(state_dim, 1, param_dim, param_dim, emb, latent, max_size=n)
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, state_dim)).astype(np.float32)
    s2 = s + 0.1 * rng.normal(size=(n, state_dim)).astype(np.float32)
    buf.add_batch(s, rng.integers(0, K, n), rng.uniform(-1, 1, (n, param_dim)), rng.uniform(-1, 1, (n, param_dim)),
                  rng.uniform(-1, 1, (n, emb)), rng.uniform(-1, 1, (n, latent)), s2, s2 - s,
                  rng.normal(size=n), np.zeros(n))
    st = [torch.as_tensor(x, dtype=torch.float32) for x in (s, rng.integers(0, K, n), rng.uniform(-1, 1, (n, param_dim)), s2 - s)]
    c_rate, recon_s = rep.get_c_rate(st[0], st[1].long(), st[2], st[3], batch_size=n, range_rate=2)
    return rep, policy, buf, c_rate, float(recon_s)


def test_fast_relabel_matches_upstream():
    rep, policy, buf, c_rate, recon_s = _setup()
    results = []
    for fast in (False, True):
        p = copy.deepcopy(policy)
        p.fast_relabel = fast
        np.random.seed(123)
        torch.manual_seed(123)
        for _ in range(4):  # includes two delayed actor updates
            p.train(buf, rep, c_rate, recon_s, batch_size=128)
        results.append(p)
    slow, fast = results
    crit = max((a - b).abs().max().item() for a, b in zip(slow.critic.parameters(), fast.critic.parameters()))
    act = max((a - b).abs().max().item() for a, b in zip(slow.actor.parameters(), fast.actor.parameters()))
    print(f"max |critic diff| = {crit:.2e}, max |actor diff| = {act:.2e}")
    assert crit < 1e-5 and act < 1e-5


def test_torch_buffer_matches_numpy_buffer():
    a = ReplayBuffer(4, 1, 3, 3, 2, 2, max_size=10)
    b = TorchReplayBuffer(4, 1, 3, 3, 2, 2, max_size=10, device="cpu")
    rng = np.random.default_rng(0)
    for _ in range(3):  # wraps around the ring buffer
        args = (rng.normal(size=(4, 4)), rng.integers(0, 3, 4), rng.normal(size=(4, 3)), rng.normal(size=(4, 3)),
                rng.normal(size=(4, 2)), rng.normal(size=(4, 2)), rng.normal(size=(4, 4)), rng.normal(size=(4, 4)),
                rng.normal(size=4), np.array([0, 0, 1, 0]))
        a.add_batch(*args)
        b.add_batch(*args)
    assert (a.size, a.ptr) == (b.size, b.ptr)
    for name in TorchReplayBuffer._FIELDS:
        assert np.allclose(getattr(a, name), getattr(b, name).numpy(), atol=1e-6), name


def test_ema_reward_scale_tracks_recent_rewards():
    p = td3.TD3(3, 2, 2, max_action=1.0, reward_norm="ema", reward_norm_decay=0.99)
    torch.manual_seed(0)
    for _ in range(2000):
        p._update_reward_stats(torch.randn(128, 1) * 1000.0)   # early, large costs
    for _ in range(2000):
        p._update_reward_stats(torch.randn(128, 1) * 1.0)      # later, small costs
    std = float(p._reward_std())
    print(f"EMA reward std after the scale drops: {std:.3f} (welford would stay near ~700)")
    assert 0.8 < std < 1.2


if __name__ == "__main__":
    test_fast_relabel_matches_upstream()
    test_torch_buffer_matches_numpy_buffer()
    test_ema_reward_scale_tracks_recent_rewards()
    print("all fast-path checks passed")
