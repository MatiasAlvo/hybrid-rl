# Policy Optimization in Hybrid Discrete-Continuous Action Spaces via Mixed Gradients

Code for learning policies in **hybrid** (discrete + continuous) action spaces, with a **mixed-gradient** training signal that **combines pathwise gradients with the score-function (likelihood-ratio) estimator**. The codebase targets **inventory-style** hybrid MDPs (e.g. fixed ordering costs) and a **switched-LQR** hybrid simulator for controlled experiments.

The primary agents maintained for experiments are **`HybridAgent`** and **`GaussianPPOAgent`** in [`src/algorithms/hybrid/agents/hybrid_agent.py`](src/algorithms/hybrid/agents/hybrid_agent.py). Other agent classes in that module exist for older ablations or baselines and **may be out of date**; use them only if you are comfortable following the code paths they exercise.

At a high level, an **agent** is the training orchestrator: it wires **policy** and **value** networks, selects which **loss components** are active (policy gradient, value, pathwise, entropy, depending on the agent), and connects to **[`FeatureRegistry`](src/features/feature_registry.py)** for building and normalizing observations and **[`RangeManager`](src/envs/inventory/range_manager.py)** for mapping network outputs into **valid continuous ranges** given the discrete structure (thresholds and sub-ranges). The training loop and optimizer stepping live in [`src/training/trainer.py`](src/training/trainer.py), using [`src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py`](src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py).

## Installation

Prerequisites: **Git**, a recent **Miniconda or Anaconda** (Conda 23+ / libmamba solver recommended), and enough disk space (CUDA PyTorch envs are large). A **NVIDIA GPU driver** matching your CUDA stack is only required if you intend to train on GPU; the conda solver can still build the env on a CPU-only machine, but training will use GPU only if PyTorch sees a device.

1. Clone the repository. Prefer **HTTPS** if you do not use GitHub SSH keys; use **SSH** if you do. Replace the URL for a fork. The directory created is the repo name (e.g. `hybrid-rl`); `cd` into that folder (or whatever name you cloned into).

```bash
# HTTPS (works without SSH keys)
git clone https://github.com/MatiasAlvo/hybrid-rl.git
cd hybrid-rl

# or SSH
# git clone git@github.com:MatiasAlvo/hybrid-rl.git && cd hybrid-rl
```

2. Create the conda environment from [`environment.yml`](environment.yml). The file lists `pytorch`, **`nvidia`** (so `pytorch-cuda` can resolve `cuda-cudart`), `conda-forge`, and `defaults`:

```bash
conda env create -f environment.yml
conda activate exp_neural
```

If `conda activate` fails with “command not found”, run `conda init bash` (or `zsh`), restart the shell, or run `source "$(conda info --base)/etc/profile.d/conda.sh"` before activating.

3. **PyTorch / CUDA:** `environment.yml` pins `pytorch-cuda=11.8`. For **CPU-only**, remove or replace the `pytorch-cuda` line and install a CPU build of PyTorch from the [PyTorch install matrix](https://pytorch.org/get-started/locally/) before or after `conda env create`, or use a separate env file. For another CUDA version, adjust `pytorch-cuda` and ensure matching PyTorch builds are available from the listed channels.

4. **Smoke check (optional):** from the repo root, with the env active:

```bash
python -c "import torch, tensordict, torchrl, gymnasium; print(torch.__version__)"
python main_run.py both fixed_costs_main hybrid_main
```

The second command runs training then evaluation; it is heavy on CPU/GPU. Use smaller `epochs` / `params_by_dataset` in the YAMLs if you only want to verify the stack starts.

### Dependencies

For a single training run via [`main_run.py`](main_run.py), you need the conda stack in `environment.yml`, in particular **Python 3.10**, **PyTorch**, **NumPy**, **PyYAML**, and **Gymnasium**. **`tensordict`** and **`torchrl`** are imported (e.g. [`src/envs/inventory/env.py`](src/envs/inventory/env.py), [`src/data/data_handling.py`](src/data/data_handling.py)) and should be treated as **required** for this project, not optional extras.

**`ray`** is used by [`sweep.py`](sweep.py) for distributed sweeps; it is **not** required for one-off `main_run.py` jobs. The same file also pulls in Jupyter, Plotly, and other tooling—see `environment.yml` for the full list.

### Weights & Biases

If a policy YAML sets `logging_params.use_wandb: true`, run `wandb login` or set `WANDB_MODE=offline` for offline logging. Note: [`src/utils/logger.py`](src/utils/logger.py) currently does **not** pass `wandb_entity` into `wandb.init`, so the `wandb_entity` field in YAML is effectively unused for entity scoping.

For training from scratch, ensure `trainer_params.load_previous_model` is `false` or that `trainer_params.load_model_path` points to a valid checkpoint.

## Project structure

| Area | Path(s) |
|------|---------|
| CLI entry | [`main_run.py`](main_run.py) |
| Setting configs | [`configs/settings/`](configs/settings/) |
| Policy / hyperparameter configs | [`configs/policies/`](configs/policies/) |
| Sweeps (optional) | [`configs/sweeps/`](configs/sweeps/), [`sweep.py`](sweep.py) |
| Hybrid agents and losses | [`src/algorithms/hybrid/agents/`](src/algorithms/hybrid/agents/), [`src/algorithms/hybrid/losses/`](src/algorithms/hybrid/losses/) |
| Optimizer wrappers | [`src/algorithms/hybrid/optimizer_wrappers/`](src/algorithms/hybrid/optimizer_wrappers/) |
| Policy networks | [`src/algorithms/common/policies/`](src/algorithms/common/policies/) |
| Features | [`src/features/feature_registry.py`](src/features/feature_registry.py) |
| Range manager | [`src/envs/inventory/range_manager.py`](src/envs/inventory/range_manager.py) |
| Hybrid inventory simulator | [`src/envs/inventory/hybrid_simulator.py`](src/envs/inventory/hybrid_simulator.py) |
| LQR hybrid simulator | [`src/envs/inventory/lqr_hybrid_simulator.py`](src/envs/inventory/lqr_hybrid_simulator.py) |
| LQR matrix preparation / cache | [`src/envs/inventory/lqr_matrix_store.py`](src/envs/inventory/lqr_matrix_store.py) (invoked from `main_run.py` via `prepare_lqr_matrices`) |
| Inventory env glue | [`src/envs/inventory/env.py`](src/envs/inventory/env.py), [`src/envs/inventory/simulator.py`](src/envs/inventory/simulator.py) |
| Data / scenarios | [`src/data/data_handling.py`](src/data/data_handling.py) |
| Training loop | [`src/training/trainer.py`](src/training/trainer.py) |
| Config helper | [`src/utils/config.py`](src/utils/config.py) |
| Logging | [`src/utils/logger.py`](src/utils/logger.py) |

## Base policy configs (starting points)

- **[`configs/policies/hybrid_main.yml`](configs/policies/hybrid_main.yml)** — minimal template for the **hybrid (mixed-gradient)** agent; intended as a **starting point for hyperparameter search (HPO)**.
- **[`configs/policies/gaussian_ppo_main.yml`](configs/policies/gaussian_ppo_main.yml)** — minimal template for **Gaussian PPO** on the same hybrid action space (PPO-style signal only).

## Reproducing reported experiments (config pairs)

These file pairs match the paper’s **joint replenishment (JRP)** fixed-cost setting and the **switched LQR** setting:

| Experiment | Setting YAML | Policy YAMLs |
|------------|----------------|--------------|
| JRP fixed costs | [`configs/settings/fixed_costs_main.yml`](configs/settings/fixed_costs_main.yml) | Hybrid: [`configs/policies/hybrid_separate_net_heterogen.yml`](configs/policies/hybrid_separate_net_heterogen.yml) — Gaussian PPO: [`configs/policies/gaussian_ppo_separate_net_heterogen.yml`](configs/policies/gaussian_ppo_separate_net_heterogen.yml) |
| Switched LQR | [`configs/settings/lqr_switched_main.yml`](configs/settings/lqr_switched_main.yml) | [`configs/policies/hybrid_separate_net_lqr.yml`](configs/policies/hybrid_separate_net_lqr.yml) and [`configs/policies/gaussian_ppo_separate_net_lqr.yml`](configs/policies/gaussian_ppo_separate_net_lqr.yml) |

## Running experiments

`main_run.py` expects **three** arguments after the script name:

1. **Mode:** `train`, `test`, or `both` (`both` runs training then evaluation on the test split).
2. **Setting stem:** basename of a file in `configs/settings/` (without `.yml`).
3. **Policy stem:** basename of a file in `configs/policies/` (without `.yml`).

Examples aligned with the reproduction table:

```bash
# Hybrid agent, JRP-style fixed-cost setting
python main_run.py both fixed_costs_main hybrid_separate_net

# Gaussian PPO, switched LQR setting
python main_run.py both lqr_switched_main gaussian_ppo_separate_net_lqr
```

If you omit the setting and policy stems, defaults are hard-coded in `main_run.py`; prefer passing stems explicitly for reproducibility.

### Switched LQR simulator (brief)

Set `problem_params.simulator_type` to `lqr_hybrid`. The discrete action selects the LQR **mode**; the continuous action has dimension `n_stores`. You can supply explicit per-mode `A`, `B`, `Q`, `R` under `problem_params.lqr`, or rely on generation and caching (see [`lqr_matrix_store.py`](src/envs/inventory/lqr_matrix_store.py)): parameters such as `lambda_u`, `seed`, and the coupling / instability knobs in [`configs/settings/lqr_switched_main.yml`](configs/settings/lqr_switched_main.yml) feed matrix generation at startup. Matrices may be cached under `configs/matrices/` unless `lqr.source_path` points to an `.npz` file. A fuller example without the `_main` suffix is [`configs/settings/lqr_switched.yml`](configs/settings/lqr_switched.yml).

---

## Policy config reference

The following documents **every key** that appears in **[`configs/policies/hybrid_main.yml`](configs/policies/hybrid_main.yml)** or **[`configs/policies/gaussian_ppo_main.yml`](configs/policies/gaussian_ppo_main.yml)**. Other policy YAMLs in the repo may add fields (for example `do_dev_every_n_epochs`); those are optional extensions beyond this minimal union.

### `trainer_params`

| Key | Meaning |
|-----|---------|
| `epochs` | Number of training epochs over the training data schedule. |
| `print_results_every_n_epochs` | How often to print training summaries to the console. |
| `save_model` | Whether to write model checkpoints. |
| `epochs_between_save` | Minimum epoch spacing between saves when saving is enabled. |
| `choose_best_model_on` | Metric used to pick the best checkpoint (e.g. `dev_loss`). |
| `load_previous_model` | If true, load weights from `load_model_path` at startup. |
| `load_model_path` | Filesystem path to a checkpoint when resuming. |
| `base_dir` | Root directory under which saved models are stored. |

### `logging_params`

| Key | Meaning |
|-----|---------|
| `use_wandb` | Enable Weights & Biases logging. |
| `wandb_project_name` | W&B project name. |
| `wandb_entity` | Entity string in YAML; not currently passed through to `wandb.init` (see Installation). |
| `exp_name` | Short experiment label for logs. |
| `env_name` | High-level environment label for logging. |
| `setting_name` | Setting label for logging (often mirrors `problem_params.setting_name`). |

### `agent_params`

| Key | Meaning |
|-----|---------|
| `agent_type` | `hybrid` or `gaussian_ppo`; must match the registered agent class. |
| `fixed_std` | *(Present in Gaussian PPO main only.)* If true, use a fixed diagonal standard deviation for continuous actions instead of a state-dependent head. |

### `optimizer_params`

| Key | Meaning |
|-----|---------|
| `learning_rate` | Base learning rate for the optimizer. |
| `anneal_lr` | If true, the trainer applies learning-rate annealing over training. |

#### `optimizer_params.ppo_params`

| Key | Meaning |
|-----|---------|
| `clip_coef` | PPO trust-region clip coefficient. |
| `gamma` | Discount factor for returns. |
| `gae_lambda` | GAE λ for advantage estimation. |
| `normalize_advantages` | If true, normalize advantages within the batch. |
| `use_gae` | If true, use generalized advantage estimation. |
| `num_epochs` | Number of PPO update epochs per rollout buffer refresh. |
| `value_function_coef` | Multiplier on the value-function loss. |
| `pathwise_coef` | Weight on the pathwise (simulator-gradient) loss when that branch is active. For **`GaussianPPOAgent`**, pathwise loss is disabled (`required_losses['pathwise']` is false), so this term is not added; the key is still present so YAML stays compatible with [`HybridWrapper`](src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py), which gates pathwise on the agent’s `required_losses`. |
| `reward_scaling_pathwise` | Reward scaling flag for the pathwise branch; inert for Gaussian PPO when pathwise is off. |
| `reward_scaling` | Global reward scaling flag for the rollout buffer / losses. |
| `buffer_periods` | Symmetric edge trim for **loss updates**: training uses only timesteps `buffer_periods : T - buffer_periods` (`effective_T = T - 2 * buffer_periods`; see `effective_slice` in [`hybrid_wrapper.py`](src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py)). Applies to tensors fed into PPO (policy, value, entropy) and, when pathwise loss is enabled, to the pathwise reward slice. |
| `max_grad_norm` | Gradient norm clip threshold (global norm). |
| `entropy_coef` | Entropy bonus coefficient. |
| `anneal_entropy_coef` | If true, linearly interpolate `entropy_coef` from its initial value toward `min_entropy_coef` over training epochs (`update_entropy_coef` in [`hybrid_wrapper.py`](src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py)). |
| `min_entropy_coef` | Endpoint of entropy annealing (`entropy_coef` moves toward this value over epochs when `anneal_entropy_coef` is true). |
| `disable_cross_term` | If true, disable the cross-term in the policy loss path in [`HybridWrapper`](src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py). Only applies to Hybrid. |

**`buffer_periods` vs. advantages:** `compute_advantages` runs over the **full** length `T` (including the terminal value bootstrap). `_prepare_training_tensors` then slices to `buffer_periods : T - buffer_periods`, so the backward GAE recurrence is defined on the full horizon, but only the **central** timesteps are flattened into the batch that receives policy/value/entropy (and pathwise) gradients—the first and last `buffer_periods` rows are dropped from that update.

### `nn_params.policy_network`

| Key | Meaning |
|-----|---------|
| `name` | Registered policy architecture (here `separate_network_policy`). |
| `hidden_layers` | Hidden layer widths for the MLP trunk. |
| `activation` | Activation function for hidden layers (e.g. `Tanh`). |
| `continuous_scale` | Scalar (or broadcastable) scale applied to continuous head outputs before squashing. |
| `continuous_shift` | Scalar shift paired with `continuous_scale`. |
| `observation_keys` | List of observation tensor keys from the feature registry fed into the policy. |
| `normalize_by_mean_demand` | If true, normalize selected features by mean demand (see policy implementation). |
| `heads.discrete` | Discrete head config: at minimum `enabled`; hybrid main also sets `activation` (e.g. `Linear`). |
| `heads.continuous` | Continuous head config: same pattern as `heads.discrete`. |

### `nn_params.value_network`

| Key | Meaning |
|-----|---------|
| `enabled` | If false, skip training a separate value network (uncommon for PPO-style runs here). |
| `input_size` | *(Hybrid main only.)* Explicit input dimension for the value MLP when set. |
| `hidden_layers` | Hidden layer widths for the value MLP. |
| `activation` | Activation for the value MLP hidden layers. |

---

## Setting config reference

### Top-level

| Key | Meaning |
|-----|---------|
| `seeds` | Named integer seeds for each randomness stream (see keys below). |
| `test_seeds` | Same key names as `seeds`, used when building the test split so test randomness can differ (e.g. different `demand` seed while keeping cost seeds aligned). |
| `wandb_config` | *(Fixed-costs main only.)* `group` and `track_params` for sweep / bookkeeping conventions. **Not read by Python training code** in this repository (no `wandb_config` references in `.py` sources); safe to omit unless an external sweep tool consumes it. |
| `sample_data_params` | Dataset construction options (below). |
| `problem_params` | Scenario definition (below). |
| `observation_params` | Observation layout (below). |
| `store_params` | Per-store sampling and dynamics defaults (below). |
| `params_by_dataset` | Per-split (`train`, `dev`, `test`) rollout sizes and horizons (below). |

#### `sample_data_params`

| Key | Meaning |
|-----|---------|
| `split_by_period` | If true, split demand traces by time period across splits instead of by sample index (used for certain real-data settings). |
| `train_periods` / `dev_periods` / `test_periods` | Period limits when splitting by period. |

#### `params_by_dataset.{train,dev,test}`

| Key | Meaning |
|-----|---------|
| `n_samples` | Number of scenarios (i.e., trajectories) in the split. |
| `batch_size` | Number of scenarios per batch. |
| `periods` | Simulation horizon length per scenario. |
| `ignore_periods` | Leading periods excluded from reported loss / metrics. |

### `problem_params`

| Key | Meaning |
|-----|---------|
| `setting_name` | String identifier for the scenario (e.g. `one_store_fixed_costs`, `lqr_switched`). |
| `n_stores` | Number of retail stores (state dimension for LQR hybrid). |
| `lost_demand` | If true, unmet demand is lost; if false, backordered. |
| `maximize_profit` | If true, maximize profit; if false, minimize cost. |
| `is_hybrid` | If true, use hybrid (discrete + continuous) action spaces and simulators that support them. |
| `discrete_features.fixed_ordering_cost` | *(Fixed-costs main only.)* `thresholds` and `values` defining the piecewise fixed ordering cost|
| `simulator_type` | *(LQR main only.)* Set to `lqr_hybrid` to use the LQR hybrid simulator. |
| `n_modes` | *(LQR main only.)* Number of discrete LQR modes for matrix generation. |
| `skip_lqr_mode_validation` | *(LQR main only.)* Passed to [`prepare_lqr_matrices`](src/envs/inventory/lqr_matrix_store.py). **When `true`**, [`_validate_lqr_modes`](src/envs/inventory/lqr_matrix_store.py) returns immediately and **does not** verify that the discrete LQR mode specification matches `n_modes` and the matrix bundle. **When `false`**, after matrices are loaded or generated, the code checks that `discrete_features.lqr_mode.values` has length `n_modes`, that `lqr_mode.thresholds` has length `n_modes` or `n_modes + 1`, and (if explicit tensors are supplied) that `lqr.A` has leading dimension `n_modes`. |

**`skip_lqr_mode_validation` vs. continuous action bounds:** This flag concerns **only** those discrete-mode / matrix **consistency** checks. It does **not** turn interval constraints on or off for the continuous control. In the broader hybrid stack, [`RangeManager`](src/envs/inventory/range_manager.py) usually maps policy outputs into **bounded intervals** that depend on the active discrete regime (piecewise thresholds in `discrete_features`). For switched-LQR experiments you may instead configure training so the continuous component is effectively **unconstrained in ℝⁿ** (subject only to how the policy samples and how the LQR dynamics use `u`); that is a modeling / range-configuration choice, **not** something this boolean changes—setting it to `true` only skips the YAML/matrix **mode-count** validation above.

| `discrete_features.lqr_mode` | *(LQR main only.)* `thresholds` and `values` for the discrete mode feature. |
| `lqr` | *(LQR main only.)* Block passed to [`prepare_lqr_matrices`](src/envs/inventory/lqr_matrix_store.py): includes `lambda_u`, `seed`, `instability`, `coupling_strength`, `b_high`, `b_low`, and optionally explicit `A`/`B`/`Q`/`R` or `source_path` in other YAML variants. |

### `observation_params`

| Key | Meaning |
|-----|---------|
| `include_warehouse_inventory` | Include warehouse inventory states in the observation when true. |
| `include_static_features.holding_costs` | Include per-store holding costs in observations. |
| `include_static_features.underage_costs` | Include shortage / underage cost parameters. |
| `include_static_features.procurement_costs` | Include procurement cost parameters. |
| `include_static_features.lead_times` | Include lead times. |
| `include_static_features.upper_bounds` | Include upper-bound features when true. |
| `include_static_features.mean` | Include mean demand (or related) static features. |
| `include_static_features.std` | Include demand standard deviation static features. |
| `demand.past_periods` | Length of past demand history in the observation. |
| `demand.period_shift` | Index shift for where past demand windows start. |
| `include_past_observations.arrivals` | Number of past in-transit arrival observations to include. |
| `include_past_observations.orders` | Number of past order observations to include. |
| `time_features_file` | Optional path to time-feature definitions. |
| `time_features` | Optional in-config list of time features. |
| `sample_features_file` | Optional path to sample-level features. |
| `sample_features` | Optional in-config list of sample features. |
| `normalize_observations` | If true, apply configured observation normalization. |

### `store_params`

#### `store_params.demand`

| Key | Meaning |
|-----|---------|
| `distribution` | Demand law (`poisson`, `normal`, `real`, etc., depending on simulator support). |
| `mean` | Baseline mean demand when not sampling per store. |
| `mean_range` | *(Fixed-costs main only.)* Range used when sampling means across stores. |
| `sample_across_stores` | Whether demand parameters vary by store according to sampling rules. |
| `expand` | Whether to broadcast scalar parameters across stores or scenarios. |
| `clip` | Whether to clip sampled demand at zero. |
| `decimals` | Decimal rounding for demand samples. |

#### `store_params.lead_time`, `holding_cost`, `underage_cost`, `procurement_cost`

| Key | Meaning |
|-----|---------|
| `value` | Fixed scalar copied everywhere when `expand` is used. |
| `vary_across_samples` | Whether the parameter is re-drawn per scenario sample. |
| `sample_across_stores` | Whether the parameter is drawn independently per store. |
| `expand` | Broadcast a single drawn value across stores or scenarios. |
| `range` | *(Fixed-costs main for costs.)* Uniform sampling interval for the parameter. |

#### `store_params.initial_inventory`

| Key | Meaning |
|-----|---------|
| `sample` | If true, randomize initial inventory. |
| `inventory_periods` | Number of inventory-related periods encoded in the state. |
| `uniform_range` | *(LQR main only.)* Uniform sampling interval for initial state components. |
| `scale_by_sqrt_dim` | *(LQR main only.)* If true, scale the uniform range by `sqrt(n_stores)` when sampling. |

### `warehouse_params`

| Key | Meaning |
|-----|---------|
| `holding_cost` | Warehouse holding cost coefficient. |
| `lead_time` | Warehouse replenishment lead time. |

### `seeds` / `test_seeds` streams

Each of `seeds` and `test_seeds` may define integers for: `underage_cost`, `holding_cost`, `procurement_cost`, `mean`, `coef_of_var`, `lead_time`, `demand`, `initial_inventory`.

## Hybrid `forward`: PS-MDPs, config ranges, and `RangeManager`

In the paper’s **PS-MDP** (piecewise-smooth MDP) view, the transition and cost are smooth **within** each region induced by discrete structural variables (e.g. whether a fixed ordering cost is paid, or which LQR mode is active), while **moving across** region boundaries introduces non-smoothness. Operationally, each discrete regime should be paired with a **valid set of continuous controls** (order quantities, LQR forces, etc.). In this codebase those regimes and their geometry are **declared in the setting YAML** under `problem_params.discrete_features`: for each named feature you supply ordered **`thresholds`** that partition the auxiliary scalar axis and **`values`** (one fewer than thresholds) that label the cells. Inventory experiments typically use `fixed_ordering_cost`; switched LQR uses `lqr_mode` (see [`configs/settings/fixed_costs_main.yml`](configs/settings/fixed_costs_main.yml) and [`configs/settings/lqr_switched_main.yml`](configs/settings/lqr_switched_main.yml)).

When `problem_params.is_hybrid` is true, [`main_run.py`](main_run.py) builds a **[`RangeManager`](src/envs/inventory/range_manager.py)** from a dict that **merges** `problem_params` with `nn_params['policy_network']` (so discrete structure and policy head options share one config object). `RangeManager` collects all threshold breakpoints, forms the **sub-range index** `n_sub_ranges`, and precomputes how each sub-range maps to feasible continuous intervals and activations (e.g. softplus vs identity tails). That object is wrapped by a **[`FeatureRegistry`](src/features/feature_registry.py)**, which the hybrid policy and value networks consume.

**[`HybridAgent.forward`](src/algorithms/hybrid/agents/hybrid_agent.py)** (simplified narrative):

1. **`prepare_inputs`** — The registry flattens / normalizes raw simulator observations into the tensor the policy sees.
2. **Policy trunk** — The network emits raw **discrete logits** and a **continuous** tensor sized for “every store × every sub-range” (the head width comes from `RangeManager.get_network_dimensions()` once ranges are known).
3. **`process_discrete_output`** — Logits become probabilities or samples; for the usual path this goes through **`range_manager.get_discrete_probabilities`**, so the discrete choice respects the same piecewise structure encoded in YAML.
4. **`process_continuous_output`** — Given the sampled **discrete action indices**, the registry feeds raw continuous outputs through **`range_manager.apply_activations`**, optional mean-demand scaling, then **`range_manager.scale_continuous_by_ranges`** with **`get_continuous_ranges()`**, so each component is squashed into the **feasible interval for the active regime** (and per-store). That is the step that **enforces feasibility** of continuous actions relative to the PS-MDP cells, rather than leaving the pre-activation ℝ outputs as physical orders. If you set `disable_continuous_scaling: true` under `policy_network` in the policy YAML, `apply_scaling` is turned off and the network output is passed through without that range mapping (useful only when you intentionally want an unconstrained intermediate representation).

5. **`compute_feature_actions_from_outputs`** — Combines discrete probabilities with the scaled continuous vector into per-feature physical actions for the simulator.
6. **Value head** — Runs on the same `processed_obs` for critic / GAE targets.

The same registry and range machinery are used by **`GaussianPPOAgent`** where continuous actions are stochastic: sampling happens before the scaling path, but **feasible executed actions** still pass through `scale_continuous_by_ranges` when scaling is enabled.

## License

MIT License

Copyright 2025 Matias Alvo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
