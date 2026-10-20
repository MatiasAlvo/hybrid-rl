"""Standalone unit tests for the HyAR port. Prints PASS/FAIL per check.

Run:
  MPLCONFIGDIR=/tmp/mplcfg PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    python test_hyar_components.py

Checks:
  1. VAE forward runs; VAE loss (recon_s + 2*recon_c + 0.5*KL) DECREASES over ~500
     steps on synthetic (state, discrete, param, next_state) data.
  2. Embedding-table nearest-neighbour decode (select_discrete_action) -> indices in [0,K).
  3. get_c_rate / z_range runs for latent dim 6 and returns finite bounds.
  4. TD3 train() runs one update on a filled ReplayBuffer without error and updates params.
  5. select_action returns (k in [0,K), x with shape (param_dim,)).
"""

import copy
import numpy as np
import torch

from src.algorithms.hyar import action_representation as ActionRepresentation_vae
from src.algorithms.hyar import td3 as P_TD3_relable
from src.algorithms.hyar import replay_buffer as hyar_utils
from src.algorithms.hyar.agent import HyARAgent
from tests.hyar.mock_env import MockPamdpEnv


RESULTS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    RESULTS.append((name, bool(cond)))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def _make_synthetic(n, state_dim, K, param_dim, seed=0):
    rng = np.random.RandomState(seed)
    s = rng.randn(n, state_dim).astype(np.float32)
    d = rng.randint(0, K, size=n).astype(np.int64)
    c = np.tanh(rng.randn(n, param_dim)).astype(np.float32) * 0.5
    # next_state deterministically depends on (s, d, c) so the VAE has something to learn
    ns = (0.9 * s + 0.05 * np.tile(c, (1, (state_dim + param_dim - 1) // param_dim))[:, :state_dim]
          + 0.05 * d.reshape(-1, 1)).astype(np.float32)
    return s, d, c, ns


# --------------------------------------------------------------------------- #
def test_vae_loss_decreases():
    state_dim, K, param_dim, latent = 6, 3, 4, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar = ActionRepresentation_vae.Action_representation(
        state_dim=state_dim, action_dim=K, parameter_action_dim=param_dim,
        reduced_action_dim=6, reduce_parameter_action_dim=latent, embed_lr=1e-3)

    s, d, c, ns = _make_synthetic(2000, state_dim, K, param_dim, seed=1)
    s_t = torch.FloatTensor(s).to(device)
    d_t = torch.LongTensor(d).to(device)
    c_t = torch.FloatTensor(c).to(device)
    ns_t = torch.FloatTensor(ns).to(device)
    delta_t = ns_t - s_t

    # forward once (smoke)
    emb = ar.get_embedding(d_t)
    recon_c, recon_s, mean, std = ar.vae(s_t, emb, c_t)
    forward_ok = (recon_c.shape == c_t.shape and recon_s.shape == s_t.shape)
    check("1a VAE forward runs (shapes match)", forward_ok,
          f"recon_c{tuple(recon_c.shape)} recon_s{tuple(recon_s.shape)}")

    bs = 256
    losses = []
    for step in range(500):
        idx = np.random.randint(0, s.shape[0], size=bs)
        vae_loss, rs, rc, kl = ar.unsupervised_loss(
            s_t[idx], d_t[idx], c_t[idx], delta_t[idx], bs, 1e-3)
        losses.append(float(vae_loss))
    first = float(np.mean(losses[:20]))
    last = float(np.mean(losses[-20:]))
    check("1b VAE loss decreases over 500 steps", last < first,
          f"mean_first20={first:.4f} -> mean_last20={last:.4f}")


def test_nn_decode():
    state_dim, K, param_dim, latent = 6, 5, 4, 6
    ar = ActionRepresentation_vae.Action_representation(
        state_dim=state_dim, action_dim=K, parameter_action_dim=param_dim,
        reduced_action_dim=6, reduce_parameter_action_dim=latent)
    # exact table rows (tanh'd) should decode back to their own index
    ok_exact = True
    for k in range(K):
        emb = ar.get_embedding(k).detach().cpu().view(1, -1)
        dec = int(ar.select_discrete_action(emb))
        ok_exact = ok_exact and (dec == k)
    check("2a NN decode recovers exact table rows", ok_exact, f"K={K}")

    # random queries -> valid indices in [0,K)
    rng = np.random.RandomState(3)
    q = torch.FloatTensor(rng.uniform(-1, 1, size=(50, 6)))
    decs = ar.select_discrete_action(q)
    decs = np.atleast_1d(np.asarray(decs))
    ok_range = bool(((decs >= 0) & (decs < K)).all())
    check("2b NN decode returns indices in [0,K)", ok_range,
          f"min={int(decs.min())} max={int(decs.max())} K={K}")


def test_c_rate():
    state_dim, K, param_dim, latent = 6, 3, 4, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar = ActionRepresentation_vae.Action_representation(
        state_dim=state_dim, action_dim=K, parameter_action_dim=param_dim,
        reduced_action_dim=6, reduce_parameter_action_dim=latent)
    s, d, c, ns = _make_synthetic(4000, state_dim, K, param_dim, seed=4)
    s_t = torch.FloatTensor(s)
    d_t = torch.LongTensor(d)
    c_t = torch.FloatTensor(c)
    delta_t = torch.FloatTensor(ns) - s_t
    c_rate, recon_s = ar.get_c_rate(s_t, d_t, c_t, delta_t, batch_size=4000, range_rate=2)
    n_dims_ok = (len(c_rate) == latent)
    all_finite = all(np.isfinite(np.asarray(cr, dtype=np.float64)).all() for cr in c_rate)
    bounds_ordered = all(cr[0] >= cr[1] for cr in c_rate)  # up >= down
    check("3 get_c_rate/z_range (latent=6) finite bounds", n_dims_ok and all_finite,
          f"n_dims={len(c_rate)} recon_s={float(recon_s):.4f} up>=down={bounds_ordered}")


def test_td3_update():
    state_dim, K, param_dim, latent, disc_emb = 6, 3, 4, 6, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar = ActionRepresentation_vae.Action_representation(
        state_dim=state_dim, action_dim=K, parameter_action_dim=param_dim,
        reduced_action_dim=disc_emb, reduce_parameter_action_dim=latent, embed_lr=1e-3)
    # pretrain VAE a little so get_c_rate is meaningful
    s, d, c, ns = _make_synthetic(3000, state_dim, K, param_dim, seed=5)
    s_t = torch.FloatTensor(s).to(device); d_t = torch.LongTensor(d).to(device)
    c_t = torch.FloatTensor(c).to(device); delta_t = torch.FloatTensor(ns).to(device) - s_t
    for _ in range(50):
        idx = np.random.randint(0, s.shape[0], size=128)
        ar.unsupervised_loss(s_t[idx], d_t[idx], c_t[idx], delta_t[idx], 128, 1e-3)
    c_rate, recon_s = ar.get_c_rate(s_t, d_t, c_t, delta_t, batch_size=s.shape[0], range_rate=2)

    policy = P_TD3_relable.TD3(
        state_dim=state_dim, discrete_action_dim=disc_emb, parameter_action_dim=latent,
        max_action=1.0, policy_noise=0.1, noise_clip=0.5, policy_freq=1)  # freq=1 -> actor updates now

    rb = hyar_utils.ReplayBuffer(state_dim=state_dim, discrete_action_dim=1,
                                 parameter_action_dim=param_dim, all_parameter_action_dim=param_dim,
                                 discrete_emb_dim=disc_emb, parameter_emb_dim=latent, max_size=5000)
    rng = np.random.RandomState(6)
    for i in range(1000):
        st = rng.randn(state_dim).astype(np.float32)
        k = rng.randint(0, K)
        x = np.tanh(rng.randn(param_dim)).astype(np.float32) * 0.5
        demb = ar.get_embedding(int(k)).detach().cpu().numpy()
        pemb = rng.uniform(-1, 1, size=latent).astype(np.float32)
        nst = (0.9 * st).astype(np.float32)
        rb.add(st, k, x, x, demb, pemb, nst, nst - st, float(-np.sum(x * x)), False)

    before = copy.deepcopy([p.detach().clone() for p in policy.critic.parameters()])
    err = None
    try:
        dr, pr = policy.train(rb, ar, c_rate, recon_s, batch_size=128)
    except Exception as e:  # noqa
        err = e
    ran_ok = err is None
    check("4a TD3 train() runs one update w/o error", ran_ok, "" if ran_ok else f"{type(err).__name__}: {err}")
    if ran_ok:
        after = [p.detach().clone() for p in policy.critic.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        check("4b TD3 update changes critic params", changed,
              f"discrete_relable_rate={float(dr):.3f} parameter_relable_rate={float(pr):.3f}")


def test_select_action():
    env = MockPamdpEnv(state_dim=6, K=3, param_dim=4, seed=7)
    agent = HyARAgent(state_dim=env.state_dim, K=env.K, param_dim=env.param_dim,
                      reduced_action_dim=6, reduce_parameter_action_dim=6, seed=7)
    agent.warmup_and_pretrain_vae(env, n_warmup_eps=200, n_vae_steps=300, verbose=False)
    state = env.reset()
    k, x = agent.select_action(state, explore=False)
    ok_k = isinstance(k, int) and (0 <= k < env.K)
    x = np.asarray(x)
    ok_x = (x.shape == (env.param_dim,)) and np.isfinite(x).all()
    check("5a select_action k in [0,K)", ok_k, f"k={k} K={env.K}")
    check("5b select_action x shape (param_dim,)", ok_x, f"x.shape={x.shape} param_dim={env.param_dim}")
    # explore path too
    k2, x2 = agent.select_action(state, explore=True)
    ok2 = (0 <= int(k2) < env.K) and np.asarray(x2).shape == (env.param_dim,)
    check("5c select_action(explore=True) valid", ok2, f"k={k2} x.shape={np.asarray(x2).shape}")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 60)
    print("HyAR port component tests")
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    for fn in (test_vae_loss_decreases, test_nn_decode, test_c_rate,
               test_td3_update, test_select_action):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            check(fn.__name__ + " (crashed)", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_total = len(RESULTS)
    print(f"SUMMARY: {n_pass}/{n_total} checks PASSED")
    for name, ok in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 60)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
