"""HyAR runner for switched LQR and the JRP, scenario-batched and sample-matched to HPO.

Fair-comparison design (see memory hyar-comparison-protocol / hyar-calibration-finding):
  * SAMPLE BUDGET = environment transitions, matched to HPO
    (HPO: 500 iters x 1024 train scenarios x 20 periods = 10.24M transition-evaluations).
  * HyAR fills its replay buffer by rolling out over the SAME 1024 training scenarios (CRN)
    and does TD3 updates at a fixed UTD (updates per transition). Rollout is ~free
    (~325k trans/s); cost is the TD3 updates (~20 ms each), so total updates = UTD * budget
    is the real compute knob. UTD is tuned once and held fixed across the sweep; we REPORT it.
  * Continuous action fairness: HyarPamdpWrapper maps HyAR's (-1,1) action to physical units
    via a config-derived action_scale (verified non-binding vs Riccati optimal |u*|), and
    normalizes states. The inner PamdpEnv stays bit-for-bit HPO-equivalent; loss/reported is
    computed in physical units by the inner env.

Logs cost-vs-transitions (headline sample axis) + wall-clock to JSON/CSV under runs/.

Example (from the repository root), with the settings used for the comparison with HPO and PPO:
  CUDA_VISIBLE_DEVICES=0 python hyar_run.py --config configs/hyar/lqr.yml --p 16 --J 2 --seed 0
  CUDA_VISIBLE_DEVICES=0 python hyar_run.py --config configs/hyar/jrp.yml --p 20 --seed 0
Without --config, the defaults are the original port's (upstream HyAR) recipe.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from src.envs.pamdp.pamdp_env import PamdpEnv
from src.algorithms.hyar.env_wrapper import HyarPamdpWrapper
from src.algorithms.hyar.agent import HyARAgent

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "hyar_runs")

# HPO's matched sample budget: 500 iters x 1024 train scenarios x 20 periods.
HPO_TRANSITIONS = 500 * 1024 * 20  # = 10_240_000


def riccati_reference(env):
    """Best single-mode Riccati cost on the eval set (optimal iff J==1). Uses inner env
    eval_policy in PHYSICAL units, so directly comparable to HyAR's loss/reported."""
    import scipy.linalg as sla
    lqr = env.problem_params["lqr"]
    A = np.array(lqr["A"]); B = np.array(lqr["B"]); Q = np.array(lqr["Q"]); R = np.array(lqr["R"])
    J = A.shape[0]
    best = None
    for m in range(J):
        P = sla.solve_discrete_are(A[m], B[m], Q[m], R[m])
        Kg = np.linalg.solve(R[m] + B[m].T @ P @ B[m], B[m].T @ P @ A[m])

        def act_fn(state_batch, Kg=Kg):
            s = state_batch.detach().cpu().numpy()
            u = -(Kg @ s.T).T
            k = np.full(s.shape[0], m, dtype=np.int64)
            return k, u

        loss = env.eval_policy(act_fn)["loss/reported"]
        best = loss if best is None else min(best, loss)
    return float(best)


def run(args):
    os.makedirs(RUNS_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setting = getattr(args, "setting", "lqr")
    if setting == "lqr":
        env = PamdpEnv(setting="lqr", n_stores=args.p, n_modes=args.J, device=device,
                       sample_seed=args.seed)
    else:  # jrp: global-binary discrete (K=2), no n_modes
        env = PamdpEnv(setting="jrp", n_stores=args.p, device=device, sample_seed=args.seed)
    wrap = HyarPamdpWrapper(env, action_scale=args.action_scale, state_scale=args.state_scale)
    if args.action_scale is None and args.action_scale_mult != 1.0:
        wrap.action_scale *= args.action_scale_mult
    print(f"[env] {env}\n[wrap] {wrap}", flush=True)

    # latent dim grows modestly with p by default; -1 never compresses the continuous action:
    # max(6, continuous dimension), 6 being upstream HyAR's usual latent size
    latent = args.latent_dim
    if latent is None:
        latent = 6 if args.p <= 16 else (8 if args.p <= 32 else 12)
    elif latent == -1:
        latent = max(6, wrap.param_dim)
    agent = HyARAgent(
        state_dim=wrap.state_dim, K=wrap.K, param_dim=wrap.param_dim,
        reduced_action_dim=args.reduced_action_dim,
        reduce_parameter_action_dim=latent,
        vae_hidden_size=args.vae_hidden_size,
        batch_size=args.batch_size, policy_freq=args.policy_freq, seed=args.seed,
        discount=args.discount, buffer_device=args.buffer_device, reward_norm=args.reward_norm,
        reward_norm_decay=args.reward_norm_decay, fast_relabel=args.fast_relabel,
        vae_kl_weight=args.vae_kl_weight, vae_persistent_optimizer=args.vae_persistent_optimizer,
        buffer_size=args.buffer_size, actor_lr=args.actor_lr, critic_lr=args.critic_lr, tau=args.tau,
        critic_layer_norm=args.critic_layer_norm, target_q_max=args.target_q_max,
        td3_hidden_size=args.td3_hidden_size, expl_noise_initial=args.expl_noise_initial,
        expl_noise=args.expl_noise, epsilon_steps=args.epsilon_steps,
    )
    agent.track_stats = agent.policy.track_stats = args.log_diagnostics
    print(f"[agent] device={agent.device} disc_emb={agent.reduced_action_dim} "
          f"latent={agent.parameter_emb_dim} batch={agent.batch_size} utd={args.utd}", flush=True)

    if setting == "lqr":
        ric = riccati_reference(env)
        print(f"[ref] best single-mode Riccati loss/reported = {ric:.5f}"
              f"{'  (OPTIMAL, J=1)' if args.J == 1 else '  (fixed-mode baseline)'}", flush=True)
    else:
        ric = float("nan")  # JRP has no closed-form optimum; compare vs HPO JRP numbers
        print("[ref] JRP: no closed-form optimum; reference = HPO JRP cost (external).", flush=True)

    # ---- reward standardization is handled ADAPTIVELY inside TD3 (running batch-std of rewards;
    # fair analog of HPO's running cost standardization, paper C.2). A fixed env-wrapper scale
    # cannot work here: S-LQR per-step cost spans ~1e4 (random) -> ~1 (optimal) over training.
    # So the wrapper reward_scale stays 1.0 and TD3.normalize_reward does the work.
    if args.reward_scale is not None:
        wrap.reward_scale = float(args.reward_scale)
    print(f"[reward] wrapper_scale={wrap.reward_scale:.1f}; TD3 running reward-std normalization="
          f"{agent.policy.normalize_reward}", flush=True)

    t0 = time.time()
    if args.warmup_batched_eps > 0:
        agent.warmup_batched_and_pretrain_vae(
            wrap, n_batched_eps=args.warmup_batched_eps, n_vae_steps=args.vae_pretrain_steps,
            vae_batch_size=args.vae_batch_size, verbose=True)
    else:
        agent.warmup_and_pretrain_vae(
            wrap, n_warmup_eps=args.warmup_eps, n_vae_steps=args.vae_pretrain_steps,
            vae_batch_size=args.vae_batch_size, verbose=True)
    print(f"[pretrain] {time.time()-t0:.1f}s recon_s={float(agent.recon_s):.5f}", flush=True)

    history = []

    def _snapshot_json(transitions):
        """Write an interim results JSON (current history/best) so a still-running job is usable."""
        best_so = min(h["loss/reported"] for h in history) if history else None
        snap = {"env": {"setting": setting, "p": args.p, "J": (args.J if setting == "lqr" else None),
                        "state_dim": wrap.state_dim, "K": wrap.K, "param_dim": wrap.param_dim},
                "meta": vars(args), "riccati_ref": ric, "transitions_so_far": int(transitions),
                "budget_transitions": args.budget_transitions,
                "best_loss_reported": best_so,
                "final_loss_reported": history[-1]["loss/reported"] if history else None,
                "history": history, "interim": True}
        tagi = (f"hyar_lqr_p{args.p}_J{args.J}_utd{args.utd}_seed{args.seed}_{int(args.stamp)}"
                if setting == "lqr" else f"hyar_jrp_p{args.p}_utd{args.utd}_seed{args.seed}_{int(args.stamp)}")
        od = getattr(args, "out_dir", None) or RUNS_DIR
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, tagi + ".json"), "w") as f:
            json.dump(snap, f, indent=2)

    def eval_cb(transitions):
        res = wrap.eval_policy_batched(agent)
        row = {
            "transitions": int(transitions),
            "loss/reported": float(res["loss/reported"]),
            "dev_loss/reported": (float(wrap.eval_policy_batched(agent, split="dev")["loss/reported"])
                                  if args.select_on_dev else None),
            "gap_vs_riccati_pct": (100.0 * (res["loss/reported"] - ric) / abs(ric)
                                   if ric == ric and ric else float("nan")),  # nan-safe (JRP)
            "wall_s": time.time() - t0,
        }
        if args.log_diagnostics:
            row["stats"] = agent.policy.pop_stats()
        history.append(row)
        gaps = (f"gap={row['gap_vs_riccati_pct']:+.1f}%" if row["gap_vs_riccati_pct"] == row["gap_vs_riccati_pct"]
                else "gap=n/a")
        dev = f"dev={row['dev_loss/reported']:.5f}  " if args.select_on_dev else ""
        print(f"[eval] trans={transitions:>10,d}  loss/reported={row['loss/reported']:.5f}  "
              f"{dev}{gaps}  ({row['wall_s']:.0f}s)", flush=True)
        if args.log_diagnostics:
            print("[stats] " + " ".join(f"{k}={v:.3g}" for k, v in row["stats"].items()), flush=True)
        # Interim JSON snapshot so a STILL-RUNNING job is usable mid-run ("report at 3M but keep
        # going"). Cheap (small JSON). Buffer/weights checkpointing dropped per user: report-and-
        # continue makes disk-resume redundant, and compressed buffer save is ~42s (too slow to loop).
        _snapshot_json(transitions)
        return row

    # ---- optional resume from a checkpoint (extend a prior run without restarting) ----
    if getattr(args, "resume_from", None):
        agent.load_state(args.resume_from, load_buffers=True)
        print(f"[resume] loaded checkpoint from {args.resume_from} "
              f"(total_it={agent.policy.total_it}, buffer_size={agent.replay_buffer.size})", flush=True)

    eval_cb(0)
    eval_every = max(1, args.budget_transitions // args.n_evals)
    wrap.train_rollout(
        agent, n_transitions=args.budget_transitions, train_step_fn=agent.train_step,
        utd=args.utd, eval_every=eval_every, eval_cb=eval_cb,
        refresh_every_eps=args.vae_refresh_ep_interval, verbose=False,
        vae_refresh_steps=args.vae_refresh_steps)
    if not history or history[-1]["transitions"] < args.budget_transitions:
        eval_cb(args.budget_transitions)

    # (Disk checkpoint save intentionally NOT run by default: we use report-and-continue, so
    #  resume-from-disk is redundant, and compressed buffer save ~42s. save_state/load_state remain
    #  available for manual use via --ckpt_dir/--resume_from if ever needed.)
    if getattr(args, "ckpt_dir", None):
        agent.save_state(args.ckpt_dir, save_buffers=not args.no_save_buffers)
        print(f"[ckpt] saved (explicit --ckpt_dir) -> {args.ckpt_dir}", flush=True)

    best = min(h["loss/reported"] for h in history)
    total_updates = int(round(args.utd * args.budget_transitions))
    log = {
        "meta": vars(args),
        "env": {"setting": setting, "p": args.p, "J": (args.J if setting == "lqr" else None),
                "state_dim": wrap.state_dim,
                "K": wrap.K, "param_dim": wrap.param_dim,
                "action_scale": wrap.action_scale, "state_scale": wrap.state_scale},
        "riccati_ref": ric,
        "budget_transitions": args.budget_transitions,
        "hpo_matched_transitions": HPO_TRANSITIONS,
        "total_td3_updates": total_updates,
        "utd": args.utd,
        "recon_s_pretrain": float(agent.recon_s),
        "final_loss_reported": history[-1]["loss/reported"],
        "best_loss_reported": best,
        # Checkpoint chosen on the dev split, scored on test (the HPO/PPO trainer's rule).
        **(_dev_selection(history) if args.select_on_dev else {}),
        "final_gap_vs_riccati_pct": history[-1]["gap_vs_riccati_pct"],
        "wall_s": time.time() - t0,
        "history": history,
    }
    tag = (f"hyar_lqr_p{args.p}_J{args.J}_utd{args.utd}_seed{args.seed}_{int(args.stamp)}"
           if setting == "lqr"
           else f"hyar_jrp_p{args.p}_utd{args.utd}_seed{args.seed}_{int(args.stamp)}")
    out_dir = getattr(args, "out_dir", None) or RUNS_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(log, f, indent=2)
    with open(os.path.join(out_dir, tag + ".csv"), "w") as f:
        f.write("transitions,loss/reported,dev_loss/reported,gap_vs_riccati_pct,wall_s\n")
        for r in history:
            f.write(f"{r['transitions']},{r['loss/reported']},{r['dev_loss/reported']},"
                    f"{r['gap_vs_riccati_pct']},{r['wall_s']}\n")
    jlbl = f"J={args.J} " if setting == "lqr" else ""
    rlbl = f"riccati={ric:.5f} " if setting == "lqr" else ""
    print(f"[done] {setting} p={args.p} {jlbl}seed={args.seed}: best={best:.5f} "
          f"final={history[-1]['loss/reported']:.5f} {rlbl}"
          f"updates={total_updates} wall={log['wall_s']:.0f}s -> {tag}.json", flush=True)
    return log


def _dev_selection(history):
    """Test cost at the evaluation with the lowest dev cost."""
    best = min(history, key=lambda h: h["dev_loss/reported"])
    return {"best_dev_loss_reported": best["dev_loss/reported"],
            "test_at_best_dev": best["loss/reported"],
            "transitions_at_best_dev": best["transitions"]}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="YAML file of argument values (see configs/hyar/); command-line flags override it")
    p.add_argument("--setting", choices=["lqr", "jrp"], default="lqr",
                   help="lqr (switched-LQR, uses --J + Riccati ref) or jrp (fixed-cost, K=2 global binary)")
    p.add_argument("--p", type=int, required=True, help="n_stores = continuous action dim")
    p.add_argument("--J", type=int, default=1, help="n_modes = #discrete actions (LQR only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--utd", type=float, default=0.03, help="TD3 updates per env transition")
    p.add_argument("--budget_transitions", type=int, default=HPO_TRANSITIONS,
                   help="env-transition sample budget (default = HPO's 10.24M)")
    p.add_argument("--n_evals", type=int, default=20, help="#eval points across training")
    p.add_argument("--reward_scale", type=float, default=None, help="override the wrapper's reward divisor (default 1; TD3 normalizes rewards itself)")
    p.add_argument("--action_scale", type=float, default=None, help="override; default LQR: 2 x p95 of the one-step greedy control, JRP: lead time x p99 demand")
    p.add_argument("--action_scale_mult", type=float, default=1.0,
                   help="multiplier on the derived default action_scale (ignored when --action_scale is set)")
    p.add_argument("--state_scale", type=float, default=None, help="override; default LQR: initial-state half-range, JRP: 1")
    p.add_argument("--latent_dim", type=int, default=None,
                   help="VAE latent dim; default grows from 6 to 12 with p, -1 means max(6, continuous dim)")
    p.add_argument("--vae_hidden_size", type=int, default=256, help="VAE MLP width (raise for robustness check)")
    p.add_argument("--reduced_action_dim", type=int, default=6, help="discrete embedding dim")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--policy_freq", type=int, default=2)
    p.add_argument("--warmup_eps", type=int, default=200)
    p.add_argument("--vae_pretrain_steps", type=int, default=5000)
    p.add_argument("--vae_batch_size", type=int, default=64)
    p.add_argument("--vae_refresh_ep_interval", type=int, default=10,
                   help="batched episodes between VAE refreshes")
    p.add_argument("--vae_refresh_steps", type=int, default=1,
                   help="VAE gradient steps per refresh (upstream HyAR: 1 per 10 single episodes)")
    p.add_argument("--warmup_batched_eps", type=int, default=0,
                   help="if > 0, warm up with this many batched random episodes over all training "
                        "scenarios instead of --warmup_eps single-scenario episodes")
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005, help="TD3 target-network update rate")
    p.add_argument("--critic_layer_norm", action="store_true", help="LayerNorm in the critic's hidden layers")
    p.add_argument("--expl_noise_initial", type=float, default=1.0, help="initial std of the latent exploration noise")
    p.add_argument("--expl_noise", type=float, default=0.1, help="final std of the latent exploration noise")
    p.add_argument("--epsilon_steps", type=int, default=1000,
                   help="batched steps (one period over all training scenarios) over which the noise decays linearly")
    p.add_argument("--td3_hidden_size", type=int, default=256,
                   help="width of the TD3 actor/critic hidden layers (upstream 256; HPO/PPO use 512)")
    p.add_argument("--target_q_max", type=float, default=None,
                   help="upper bound on TD targets; 0 is valid because rewards are negative costs")
    p.add_argument("--buffer_size", type=int, default=1_000_000,
                   help="TD3 replay buffer capacity (upstream 1e6, larger than its runs, so nothing is discarded)")
    p.add_argument("--buffer_device", type=str, default=None,
                   help="keep replay buffers as float32 tensors on this device (e.g. cuda)")
    p.add_argument("--reward_norm", choices=["welford", "ema", "none"], default="welford",
                   help="TD3 reward scaling: std of all rewards so far (welford), moving std (ema), or none")
    p.add_argument("--reward_norm_decay", type=float, default=0.999)
    p.add_argument("--fast_relabel", action="store_true",
                   help="run the relabeling step on the device (same computation as upstream)")
    p.add_argument("--vae_kl_weight", type=float, default=0.5,
                   help="weight of the VAE KL term (upstream 0.5; lower values avoid posterior collapse at high p)")
    p.add_argument("--vae_persistent_optimizer", action="store_true",
                   help="keep one Adam optimizer for the VAE instead of rebuilding it at every step (upstream)")
    p.add_argument("--log_diagnostics", action="store_true",
                   help="record critic loss, Q scale, reward scale and latent/action saturation at each evaluation")
    p.add_argument("--select_on_dev", action="store_true",
                   help="also evaluate on the dev split and report test cost at the best-dev evaluation")
    p.add_argument("--stamp", type=float, default=0, help="timestamp tag (pass from shell)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="output dir for JSON/CSV (default runs/; use a subdir for probes to avoid "
                        "contaminating the main sweep table)")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="if set, save resumable state here at end (and every --ckpt_every transitions)")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="checkpoint every N transitions during training (0=only at end)")
    p.add_argument("--no_save_buffers", action="store_true",
                   help="checkpoint weights+stats only, skip replay buffers (fast, but resume is warm-start)")
    p.add_argument("--resume_from", type=str, default=None, help="ckpt_dir to resume/extend from")
    return p


def parse_args(argv=None):
    """Parse the command line, taking defaults from --config when one is given."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    config = pre.parse_known_args(argv)[0].config
    parser = build_parser()
    if config:
        import yaml
        with open(config) as f:
            values = yaml.safe_load(f) or {}
        unknown = sorted(set(values) - {a.dest for a in parser._actions})
        if unknown:
            parser.error(f"unknown keys in {config}: {unknown}")
        parser.set_defaults(**values)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
