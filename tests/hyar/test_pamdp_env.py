"""
test_pamdp_env.py -- self-test for the PAMDP adapter (src/envs/pamdp/pamdp_env.py + scenarios.py).

Run from the repository root, either through pytest or as a script that prints every check:

    python -m pytest tests/hyar
    python -m tests.hyar.test_pamdp_env

Checks (all must PASS):
  1. Dynamics fidelity (JRP + LQR): a FIXED deterministic policy driven through PamdpEnv vs
     an INDEPENDENT hand-rolled loop that calls sim.step directly with the verified
     action_dict recipes -> per-step cost + state (feature vector AND raw store_inventories)
     match to atol 1e-3.
  2. Metric fidelity: eval_policy's 'loss/reported' equals
     epoch_loss_to_report / (n_samples * (periods - ignore_periods) * n_stores), and matches
     an independent batched recomputation.
  3. Scenario determinism (CRN): same scenario_idx twice -> identical trajectory;
     JRP train `seeds` vs test `test_seeds` -> different demand.
  4. Feasibility: JRP order qty >= 0 always; JRP fixed cost = 64*p exactly iff ordering;
     LQR control can be negative (unconstrained).
  5. Construction at larger dims (JRP p=20; LQR p=16, J=2) without error.
"""

import os
import sys

import numpy as np
import torch

from src.envs.pamdp.pamdp_env import PamdpEnv       # noqa: E402
from src.envs.pamdp import scenarios                 # noqa: E402
from src.features.feature_registry import FeatureRegistry           # noqa: E402
from src.envs.inventory.hybrid_simulator import HybridSimulator     # noqa: E402
from src.envs.inventory.lqr_hybrid_simulator import LqrHybridSimulator  # noqa: E402

ATOL = 1e-3
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append(bool(ok))
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  --  {detail}"
    print(line)


# --------------------------------------------------------------------------------------- #
# Fixed deterministic policies (pure functions of the state vector, batched-friendly).
# --------------------------------------------------------------------------------------- #
def jrp_policy(state):
    """state: [dim] or [B,dim]. k=order iff mean(state)<0.8 ; x = fixed 0.3 vector."""
    s = torch.as_tensor(state, dtype=torch.float32)
    single = s.dim() == 1
    if single:
        s = s.unsqueeze(0)
    B = s.shape[0]
    k = (s.mean(dim=-1) < 0.8).long()                       # [B]
    x = 0.3 * torch.ones(B, PARAM_DIM_JRP, dtype=torch.float32)
    if single:
        return int(k[0].item()), x[0]
    return k, x


def lqr_policy(state):
    """state: [dim]==raw x, or [B,dim]. mode = int(mean(x)>0); u = -0.2*x (can be negative)."""
    s = torch.as_tensor(state, dtype=torch.float32)
    single = s.dim() == 1
    if single:
        s = s.unsqueeze(0)
    k = (s.mean(dim=-1) > 0).long()
    u = -0.2 * s
    if single:
        return int(k[0].item()), u[0]
    return k, u


# --------------------------------------------------------------------------------------- #
# Independent, hand-rolled action_dict builders (raw recipe -- NOT bridge code).
# --------------------------------------------------------------------------------------- #
def hand_action_dict_jrp(k, x, K_fix, device):
    import torch.nn.functional as F
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
    B = x_t.shape[0]
    do = torch.as_tensor(k, dtype=torch.float32, device=device).view(B)
    qty = F.softplus(x_t) * do.unsqueeze(1)
    rp = torch.zeros(B, 1, 2, device=device)
    rp[:, 0, 0] = 1.0 - do
    rp[:, 0, 1] = do
    vals = torch.tensor([0.0, float(K_fix)], device=device)
    return {"feature_actions": {"total_action": qty,
                                "fixed_ordering_cost": {"range_probs": rp, "values": vals}}}


def hand_action_dict_lqr(k, x, device):
    u = torch.as_tensor(x, dtype=torch.float32, device=device)
    if u.dim() == 1:
        u = u.unsqueeze(0)
    B = u.shape[0]
    mode = torch.as_tensor(k, dtype=torch.long, device=device).view(B)
    return {"feature_actions": {"total_action": u}, "discrete_action_indices": mode}


def fresh_sim_and_fr(env):
    """Build an independent simulator + FeatureRegistry mirroring the env's config."""
    fr = FeatureRegistry(env.nn_params, env.range_manager)
    cls = LqrHybridSimulator if env.is_lqr else HybridSimulator
    sim = cls(fr, model=None, device=env.device)
    return sim, fr


def hand_state(obs, sim, fr):
    obs_int = {k: v for k, v in obs.items()}
    obs_int["internal_data"] = sim._internal_data
    return fr.prepare_inputs(obs_int, update_ewma=True)


# --------------------------------------------------------------------------------------- #
# 1. DYNAMICS FIDELITY
# --------------------------------------------------------------------------------------- #
def check_dynamics(env, policy, hand_ad_fn, label, scenario_idx=0):
    device = env.device
    periods = env.periods

    # ---- Path A: through PamdpEnv --------------------------------------------------------
    a_costs, a_states, a_raw = [], [], []
    s = env.reset(scenario_idx)
    a_states.append(np.asarray(s, dtype=np.float32))
    a_raw.append(env._obs["store_inventories"][:, :, 0].detach().cpu().numpy().copy())
    k_seen = set()
    for _ in range(periods):
        k, x = policy(s)
        k_seen.add(int(k if np.isscalar(k) else int(k)))
        s, r, done, info = env.step(k, x)
        a_costs.append(info["cost"])
        a_states.append(np.asarray(s, dtype=np.float32))
        a_raw.append(env._obs["store_inventories"][:, :, 0].detach().cpu().numpy().copy())

    # ---- Path B: independent hand-rolled loop calling sim.step directly ------------------
    sim, fr = fresh_sim_and_fr(env)
    data = env._slice_data(env._train_data, [scenario_idx])
    obs, _ = sim.reset(periods, env.problem_params, data, env.observation_params)
    b_costs, b_states, b_raw = [], [], []
    s_b = hand_state(obs, sim, fr)
    b_states.append(s_b.detach().cpu().numpy().reshape(-1).astype(np.float32))
    b_raw.append(obs["store_inventories"][:, :, 0].detach().cpu().numpy().copy())
    for _ in range(periods):
        k, x = policy(s_b.reshape(-1))
        if env.is_lqr:
            ad = hand_ad_fn(k, x, device)
        else:
            ad = hand_ad_fn(k, x, env.K_fix, device)
        obs, cost, _term, _, _ = sim.step(obs, ad)
        b_costs.append(float(cost.reshape(-1)[0].item()))
        s_b = hand_state(obs, sim, fr)
        b_states.append(s_b.detach().cpu().numpy().reshape(-1).astype(np.float32))
        b_raw.append(obs["store_inventories"][:, :, 0].detach().cpu().numpy().copy())

    a_costs, b_costs = np.array(a_costs), np.array(b_costs)
    cost_err = float(np.max(np.abs(a_costs - b_costs)))
    state_err = float(max(np.max(np.abs(a - b)) for a, b in zip(a_states, b_states)))
    raw_err = float(max(np.max(np.abs(a - b)) for a, b in zip(a_raw, b_raw)))

    ok = cost_err <= ATOL and state_err <= ATOL and raw_err <= ATOL
    check(
        f"dynamics fidelity [{label}]",
        ok,
        f"max|dcost|={cost_err:.2e} max|dstate|={state_err:.2e} "
        f"max|d inv|={raw_err:.2e} (periods={periods}, k_seen={sorted(k_seen)})",
    )


# --------------------------------------------------------------------------------------- #
# 2. METRIC FIDELITY
# --------------------------------------------------------------------------------------- #
def check_metric_fidelity(env, policy, hand_ad_fn, label):
    res = env.eval_policy(policy)
    n = res["n_samples"]
    P = res["periods"]
    ig = res["ignore_periods"]
    ns = res["n_stores"]

    # (a) the returned scalar equals the exact trainer.py normalization of its own sum.
    expected = res["sum_reported"] / (n * (P - ig) * ns)
    norm_ok = abs(expected - res["loss/reported"]) <= 1e-9
    check(
        f"metric normalization identity [{label}]",
        norm_ok,
        f"loss/reported={res['loss/reported']:.6f} == sum_reported/(n*(P-ig)*ns)={expected:.6f} "
        f"[n={n}, P={P}, ignore={ig}, n_stores={ns}]",
    )

    # (b) independent batched recomputation of epoch_loss_to_report.
    sim, fr = fresh_sim_and_fr(env)
    data = {k: v.clone() for k, v in env._eval_data.items()}
    obs, _ = sim.reset(P, env.problem_params, data, env.observation_params)
    epoch_loss_to_report = 0.0
    epoch_loss_total = 0.0
    with torch.no_grad():
        for t in range(P):
            state = hand_state(obs, sim, fr)
            k, x = policy(state)
            if env.is_lqr:
                ad = hand_ad_fn(k, x, env.device)
            else:
                ad = hand_ad_fn(k, x, env.K_fix, env.device)
            obs, cost, _term, _, _ = sim.step(obs, ad)
            c = float(cost.sum().item())
            epoch_loss_total += c
            if t >= ig:
                epoch_loss_to_report += c
    indep_reported = epoch_loss_to_report / (n * (P - ig) * ns)
    indep_total = epoch_loss_total / (n * P * ns)
    match_ok = (abs(indep_reported - res["loss/reported"]) <= ATOL and
                abs(indep_total - res["loss/total"]) <= ATOL)
    check(
        f"metric fidelity vs independent rollout [{label}]",
        match_ok,
        f"reported env={res['loss/reported']:.6f} indep={indep_reported:.6f} | "
        f"total env={res['loss/total']:.6f} indep={indep_total:.6f}",
    )


# --------------------------------------------------------------------------------------- #
# 3. SCENARIO DETERMINISM (CRN)
# --------------------------------------------------------------------------------------- #
def check_crn(env, policy, label, scenario_idx=1):
    def rollout(idx):
        costs, states = [], []
        s = env.reset(idx)
        states.append(np.asarray(s, dtype=np.float32))
        for _ in range(env.periods):
            k, x = policy(s)
            s, r, done, info = env.step(k, x)
            costs.append(info["cost"])
            states.append(np.asarray(s, dtype=np.float32))
        return np.array(costs), states

    c1, st1 = rollout(scenario_idx)
    c2, st2 = rollout(scenario_idx)
    same = (np.max(np.abs(c1 - c2)) == 0.0 and
            max(np.max(np.abs(a - b)) for a, b in zip(st1, st2)) == 0.0)
    check(f"CRN: same scenario_idx -> identical trajectory [{label}]", same,
          f"max|dcost|={np.max(np.abs(c1 - c2)):.2e}")

    # different index -> generally different (sanity, not a hard identity)
    c3, _ = rollout((scenario_idx + 1) % env.n_train)
    diff = np.max(np.abs(c1 - c3)) > 0.0
    check(f"CRN: different scenario_idx -> different trajectory [{label}]", diff,
          f"max|dcost|={np.max(np.abs(c1 - c3)):.2e}")


def check_jrp_seed_difference():
    """JRP: train `seeds` vs test `test_seeds` produce different demand traces."""
    cfg = scenarios.prepare_setting_config("jrp")
    train = scenarios.make_scenarios("jrp", "train", cfg=cfg)
    test = scenarios.make_scenarios("jrp", "test", cfg=cfg)
    n = min(train["demands"].shape[0], test["demands"].shape[0])
    T = min(train["demands"].shape[2], test["demands"].shape[2])
    d_train = train["demands"][:n, :, :T]
    d_test = test["demands"][:n, :, :T]
    differ = not torch.allclose(d_train, d_test)
    maxdiff = float((d_train - d_test).abs().max().item())
    check("CRN: JRP train seeds != test_seeds demand", differ, f"max|d demand|={maxdiff:.3f}")


# --------------------------------------------------------------------------------------- #
# 4. FEASIBILITY
# --------------------------------------------------------------------------------------- #
def check_feasibility_jrp(env):
    device = env.device
    p = env.n_stores
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((7, p)) * 5.0, dtype=torch.float32, device=device)

    # order qty >= 0 always (softplus), for both ordering and non-ordering.
    ad_order = env._build_action_dict(torch.ones(7, dtype=torch.long, device=device), x)
    ad_noord = env._build_action_dict(torch.zeros(7, dtype=torch.long, device=device), x)
    qty_o = ad_order["feature_actions"]["total_action"]
    qty_n = ad_noord["feature_actions"]["total_action"]
    check("feasibility: JRP order qty >= 0 (softplus)", bool((qty_o >= 0).all().item()),
          f"min qty (ordering)={float(qty_o.min().item()):.4f}")
    check("feasibility: JRP qty == 0 when not ordering", bool((qty_n == 0).all().item()),
          f"max qty (not ordering)={float(qty_n.max().item()):.4f}")

    # fixed cost = 64*p exactly iff ordering (recompute exactly like hybrid_simulator).
    def fixed_cost(ad):
        rp = ad["feature_actions"]["fixed_ordering_cost"]["range_probs"]
        vals = ad["feature_actions"]["fixed_ordering_cost"]["values"]
        return (rp * vals.unsqueeze(0)).sum(dim=(-1, -2))

    fc_o = fixed_cost(ad_order)
    fc_n = fixed_cost(ad_noord)
    expect = 64.0 * p
    ok_scale = abs(env.K_fix - expect) < 1e-9
    ok_order = bool(torch.allclose(fc_o, torch.full_like(fc_o, expect), atol=1e-4))
    ok_noord = bool((fc_n == 0).all().item())
    check("feasibility: JRP K_fix == 64*p", ok_scale, f"K_fix={env.K_fix} expected={expect}")
    check("feasibility: JRP fixed cost == 64*p iff ordering", ok_order and ok_noord,
          f"fixed(order)={float(fc_o[0].item()):.1f}, fixed(no-order)={float(fc_n[0].item()):.1f}")


def check_feasibility_lqr(env):
    device = env.device
    p = env.n_stores
    # negative controls must pass through unchanged (unconstrained).
    x = -1.5 * torch.ones(5, p, dtype=torch.float32, device=device)
    ad = env._build_action_dict(torch.zeros(5, dtype=torch.long, device=device), x)
    u = ad["feature_actions"]["total_action"]
    ok = bool((u < 0).all().item()) and bool(torch.allclose(u, x, atol=1e-6))
    check("feasibility: LQR control can be negative (unconstrained)", ok,
          f"min u={float(u.min().item()):.3f} (== input {float(x.min().item()):.3f})")


# --------------------------------------------------------------------------------------- #
# 5. CONSTRUCTION AT LARGER DIMS
# --------------------------------------------------------------------------------------- #
def check_large_construction():
    try:
        big_jrp = PamdpEnv("jrp", n_stores=20)
        ok_jrp = (big_jrp.K == 2 and big_jrp.param_dim == 20 and
                  big_jrp.state_dim == 5 * 20 + 2)
        check("construct JRP p=20", ok_jrp,
              f"K={big_jrp.K}, param_dim={big_jrp.param_dim}, state_dim={big_jrp.state_dim}")
    except Exception as e:  # noqa: BLE001
        check("construct JRP p=20", False, f"exception: {e!r}")

    try:
        big_lqr = PamdpEnv("lqr", n_stores=16, n_modes=2)
        ok_lqr = (big_lqr.K == 2 and big_lqr.param_dim == 16 and big_lqr.state_dim == 16)
        check("construct LQR p=16 J=2", ok_lqr,
              f"K={big_lqr.K}, param_dim={big_lqr.param_dim}, state_dim={big_lqr.state_dim}")
    except Exception as e:  # noqa: BLE001
        check("construct LQR p=16 J=2", False, f"exception: {e!r}")


# --------------------------------------------------------------------------------------- #
# pytest entry points: the checks main() runs, each test failing if any of its checks fail.
# --------------------------------------------------------------------------------------- #
def _run_checks(fn, *args):
    start = len(_RESULTS)
    fn(*args)
    failed = _RESULTS[start:].count(False)
    assert failed == 0, f"{fn.__name__}: {failed} check(s) failed (see captured output)"


def _envs():
    global PARAM_DIM_JRP
    if not hasattr(_envs, "cache"):
        torch.manual_seed(0)
        np.random.seed(0)
        _envs.cache = (PamdpEnv("jrp"), PamdpEnv("lqr"))
        PARAM_DIM_JRP = _envs.cache[0].param_dim
    return _envs.cache


def test_dynamics():
    env_jrp, env_lqr = _envs()
    _run_checks(check_dynamics, env_jrp, jrp_policy, hand_action_dict_jrp, "JRP")
    _run_checks(check_dynamics, env_lqr, lqr_policy, hand_action_dict_lqr, "LQR")


def test_metric_fidelity():
    env_jrp, env_lqr = _envs()
    _run_checks(check_metric_fidelity, env_jrp, jrp_policy, hand_action_dict_jrp, "JRP")
    _run_checks(check_metric_fidelity, env_lqr, lqr_policy, hand_action_dict_lqr, "LQR")


def test_crn():
    env_jrp, env_lqr = _envs()
    _run_checks(check_crn, env_jrp, jrp_policy, "JRP")
    _run_checks(check_crn, env_lqr, lqr_policy, "LQR")
    _run_checks(check_jrp_seed_difference)


def test_feasibility():
    env_jrp, env_lqr = _envs()
    _run_checks(check_feasibility_jrp, env_jrp)
    _run_checks(check_feasibility_lqr, env_lqr)


def test_large_construction():
    _run_checks(check_large_construction)


# --------------------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    global PARAM_DIM_JRP
    print("=" * 88)
    print("PAMDP adapter self-test")
    print("=" * 88)

    print("\n[build] JRP p=5 (K=2, horizon 100, ignore 16)")
    env_jrp = PamdpEnv("jrp")
    print("       ", repr(env_jrp))
    PARAM_DIM_JRP = env_jrp.param_dim

    print("\n[build] LQR p=3 J=2 (horizon 20, ignore 0)")
    env_lqr = PamdpEnv("lqr")
    print("       ", repr(env_lqr))

    print("\n-- 1. Dynamics fidelity ------------------------------------------------------")
    check_dynamics(env_jrp, jrp_policy, hand_action_dict_jrp, "JRP")
    check_dynamics(env_lqr, lqr_policy, hand_action_dict_lqr, "LQR")

    print("\n-- 2. Metric fidelity --------------------------------------------------------")
    check_metric_fidelity(env_jrp, jrp_policy, hand_action_dict_jrp, "JRP")
    check_metric_fidelity(env_lqr, lqr_policy, hand_action_dict_lqr, "LQR")

    print("\n-- 3. Scenario determinism (CRN) ---------------------------------------------")
    check_crn(env_jrp, jrp_policy, "JRP")
    check_crn(env_lqr, lqr_policy, "LQR")
    check_jrp_seed_difference()

    print("\n-- 4. Feasibility ------------------------------------------------------------")
    check_feasibility_jrp(env_jrp)
    check_feasibility_lqr(env_lqr)

    print("\n-- 5. Construction at larger dims --------------------------------------------")
    check_large_construction()

    print("\n" + "=" * 88)
    passed = sum(_RESULTS)
    total = len(_RESULTS)
    verdict = "ALL PASS" if passed == total else "FAILURES PRESENT"
    print(f"SUMMARY: {passed}/{total} checks passed -- {verdict}")
    print("=" * 88)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
