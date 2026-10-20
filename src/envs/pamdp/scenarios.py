"""
scenarios.py -- build hybrid-rl Scenario data EXACTLY as main_run.py does, for the
HyAR-vs-HPO PAMDP comparison (Exp 2).

Two settings are supported:
  * 'jrp' -> configs/settings/fixed_costs_main.yml   (joint replenishment, fixed costs)
  * 'lqr' -> configs/settings/lqr_switched_main.yml  (switched-LQR)

Split semantics reproduce main_run.run_training for synthetic (non-real) demand
(sample_data_params['split_by_period'] == False), i.e. the branch at main_run.py:334-367:

    train + dev come from ONE Scenario built with `seeds`,
        num_samples = train.n_samples + dev.n_samples,
        then DatasetCreator.split_by_sample_index(scenario, sample_index_for_split=dev.n_samples):
            dev  = data[:dev.n_samples]      (first rows)
            train= data[dev.n_samples:]      (remaining rows)
    test comes from a SEPARATE Scenario built with `test_seeds`, num_samples = test.n_samples.

Everything is deterministic given the YAML seeds (Scenario reseeds np.random per feature).

Constraints honoured:
  * No file in the repository is modified. Setting configs are deep-copied and
    overridden IN MEMORY only (never written back to YAML).
  * For LQR we LOAD the shipped configs/matrices/*.npz (the exact matrices HPO uses) and
    inject them into the config so prepare_lqr_matrices() returns immediately and never
    regenerates/writes any file under configs/.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, Optional, Tuple

import yaml

from src.data.data_handling import Scenario, DatasetCreator
from src.envs.inventory.lqr_matrix_store import (
    prepare_lqr_matrices,
    load_matrices_from_npz,
    _matrix_filename,
    _default_matrix_store_dir,
)

# ----------------------------------------------------------------------------- #
# Setting registry: logical name -> (setting yaml stem, default HPO policy stem)
# The default policy is the HPO policy whose FeatureRegistry defines the state the
# HPO agent sees. PamdpEnv feeds HyAR the SAME state.
# ----------------------------------------------------------------------------- #
SETTING_REGISTRY = {
    "jrp": {
        "setting_stem": "fixed_costs_main",
        "policy_stem": "hybrid_separate_net_heterogen",
    },
    "lqr": {
        "setting_stem": "lqr_switched_main",
        "policy_stem": "hybrid_separate_net_lqr",
    },
}


def _configs_dir() -> str:
    # src/envs/pamdp/scenarios.py -> repository root is three levels up.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "configs")


def _resolve_setting_stem(setting: str) -> str:
    if setting in SETTING_REGISTRY:
        return SETTING_REGISTRY[setting]["setting_stem"]
    return setting  # assume caller passed an explicit yaml stem


def resolve_policy_stem(setting: str, policy_stem: Optional[str]) -> str:
    if policy_stem is not None:
        return policy_stem
    if setting in SETTING_REGISTRY:
        return SETTING_REGISTRY[setting]["policy_stem"]
    raise ValueError(f"No default policy known for setting '{setting}'; pass policy_stem.")


def load_setting_config(setting: str) -> Dict:
    """Load a setting YAML (deep-copyable dict). Does NOT mutate any file."""
    stem = _resolve_setting_stem(setting)
    path = os.path.join(_configs_dir(), "settings", f"{stem}.yml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_policy_config(setting: str, policy_stem: Optional[str] = None) -> Dict:
    stem = resolve_policy_stem(setting, policy_stem)
    path = os.path.join(_configs_dir(), "policies", f"{stem}.yml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _scale_fixed_cost_by_stores(setting_config: Dict) -> None:
    """
    In-memory replica of main_run.scale_fixed_cost_by_stores (main_run.py:78-101):
    multiply fixed_ordering_cost.values by n_stores (64 -> 64*p), idempotently.
    """
    problem_params = setting_config.get("problem_params", {})
    n_stores = problem_params.get("n_stores", 1)
    fc = problem_params.get("discrete_features", {}).get("fixed_ordering_cost", {})
    if not fc:
        return
    if not fc.get("scale_by_n_stores", True):
        return
    if fc.get("_scaled_by_n_stores"):
        return
    values = fc.get("values")
    if not isinstance(values, list):
        return
    scaled = [(v * n_stores) if isinstance(v, (int, float)) else v for v in values]
    if scaled != values:
        fc["values"] = scaled
        fc["_scaled_by_n_stores"] = True


def _inject_cached_lqr_matrices(problem_params: Dict) -> None:
    """
    For LQR: load the SHIPPED matrices from configs/matrices/*.npz and place
    them in problem_params['lqr'] BEFORE prepare_lqr_matrices runs. This guarantees
    prepare_lqr_matrices() takes its early-return branch (all of A/B/Q/R present) and
    therefore never generates or *writes* a file under configs/. It also guarantees we
    use the exact matrices HPO trained/tested against (which were generated with the
    stable-hash seed, NOT the config's seed:0 -- verified ground truth).
    """
    if problem_params.get("simulator_type") != "lqr_hybrid":
        return
    lqr = problem_params.get("lqr", {})
    if all(k in lqr for k in ("A", "B", "Q", "R")):
        return  # already populated

    n_stores = problem_params["n_stores"]
    n_modes = problem_params["n_modes"]
    lambda_u = lqr.get("lambda_u", 0.1)
    instability = lqr.get("instability", 1.1)
    coupling = lqr.get("coupling_strength", 0.05)
    b_high = lqr.get("b_high", 1.0)
    b_low = lqr.get("b_low", 0.2)

    fname = _matrix_filename(n_stores, n_modes, lambda_u, instability, coupling, b_high, b_low)
    path = _default_matrix_store_dir() / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Cached LQR matrix file not found for p={n_stores}, J={n_modes}: {path}. "
            "Refusing to regenerate (would write into configs/matrices/). Pre-generate it via "
            "src/envs/inventory/lqr_matrix_store.py first."
        )
    mats = load_matrices_from_npz(str(path))
    problem_params["lqr"] = {**lqr, **mats}


def prepare_setting_config(
    setting: str,
    n_stores: Optional[int] = None,
    n_modes: Optional[int] = None,
) -> Dict:
    """
    Return a fully-prepared, deep-copied setting config:
      * n_stores / n_modes overridden in-memory (for p/J sweeps) -- YAML never touched,
      * fixed_ordering_cost scaled by n_stores (JRP),
      * shipped LQR matrices injected + validated (LQR).
    """
    cfg = copy.deepcopy(load_setting_config(setting))
    pp = cfg["problem_params"]
    if n_stores is not None:
        pp["n_stores"] = int(n_stores)
    if n_modes is not None and "n_modes" in pp:
        pp["n_modes"] = int(n_modes)

    _scale_fixed_cost_by_stores(cfg)
    _inject_cached_lqr_matrices(pp)
    # Validates mode counts / returns immediately since A,B,Q,R now present.
    prepare_lqr_matrices(pp)
    return cfg


def _build_scenario(cfg: Dict, num_samples: int, periods: int, seeds: Dict) -> Scenario:
    return Scenario(
        periods=periods,
        problem_params=cfg["problem_params"],
        store_params=cfg["store_params"],
        warehouse_params=cfg["warehouse_params"],
        echelon_params=cfg["echelon_params"],
        num_samples=num_samples,
        observation_params=cfg["observation_params"],
        seeds=seeds,
    )


def make_scenarios(
    setting: str,
    split: str,
    n_stores: Optional[int] = None,
    n_modes: Optional[int] = None,
    cfg: Optional[Dict] = None,
) -> Dict:
    """
    Build the repo Scenario exactly as main_run does and return the `.get_data()` tensor
    dict for the requested split.

    Args:
        setting : 'jrp' | 'lqr' (or an explicit setting yaml stem).
        split   : 'train' | 'dev' | 'test'.
        n_stores/n_modes : optional in-memory overrides (p / J sweeps).
        cfg     : optional pre-prepared setting config (from prepare_setting_config);
                  if given, n_stores/n_modes are ignored (assumed already applied).

    Returns:
        dict[str, torch.Tensor]  (Scenario.get_data() for that split).
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be train|dev|test, got {split!r}")

    if cfg is None:
        cfg = prepare_setting_config(setting, n_stores=n_stores, n_modes=n_modes)

    pbd = cfg["params_by_dataset"]
    sample_data = cfg["sample_data_params"]
    if sample_data.get("split_by_period", False):
        # Not used by fixed_costs_main / lqr_switched_main (both synthetic, split_by_period=False).
        raise NotImplementedError(
            "split_by_period=True path is not needed for the JRP/LQR PAMDP settings."
        )

    if split in ("train", "dev"):
        # ONE Scenario for train+dev, using `seeds`; split by sample index.
        num_samples = pbd["train"]["n_samples"] + pbd["dev"]["n_samples"]
        scenario = _build_scenario(cfg, num_samples, pbd["train"]["periods"], cfg["seeds"])
        train_data, dev_data = DatasetCreator().split_by_sample_index(
            scenario, sample_index_for_split=pbd["dev"]["n_samples"]
        )
        return dev_data if split == "dev" else train_data

    # split == 'test' : separate Scenario with `test_seeds`.
    test_scenario = _build_scenario(
        cfg, pbd["test"]["n_samples"], pbd["test"]["periods"], cfg["test_seeds"]
    )
    return test_scenario.get_data()


def split_runtime_params(cfg: Dict, split: str) -> Tuple[int, int, int]:
    """Return (periods, ignore_periods, n_samples) for a split from params_by_dataset."""
    pbd = cfg["params_by_dataset"][split]
    return pbd["periods"], pbd["ignore_periods"], pbd["n_samples"]
