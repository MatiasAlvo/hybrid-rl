# HyAR baseline: provenance and license

`action_representation.py`, `td3.py`, `replay_buffer.py` and `nn_utils.py` are adapted from the
reference implementation of HyAR (Li et al., "HyAR: Addressing Discrete-Continuous Action
Reinforcement Learning via Hybrid Action Representation", ICLR 2022),
https://github.com/TJU-DRL-LAB/self-supervised-rl, commit `ece95621b8c49f154f96cf7d395b95362a3b3d4e`,
files `HyAR/embedding/ActionRepresentation_vae.py`, `HyAR/agents/P_TD3_relable.py`,
`HyAR/HyAR_RL/utils.py` and `HyAR/embedding/Utils/utils.py`.

That code is distributed under the Apache License 2.0, included here as `LICENSE`. (The upstream
README states MIT; the repository's LICENSE file is Apache 2.0.)

## Changes from upstream

- `action_representation.py`: the delta-state head is scaled by `delta_scale` (default 1, the
  upstream behaviour); the VAE hidden size is configurable; `F.mse_loss` uses `reduction='mean'`
  for current PyTorch; `z_range` is vectorized and handles any latent size (identical results for
  the sizes 3, 4, 6, 8, 10, 12 and 16 that upstream supports); the KL weight (upstream 0.5) is an
  option, and so is keeping one Adam optimizer across training calls (upstream rebuilds it each call).
- `td3.py`: rewards inside the critic target are divided by a running standard deviation
  (`welford`, or an exponential moving estimate with `ema`); gradient inversion is vectorized
  (identical results); an optional relabeling path runs on the device without numpy round trips;
  learning rates, the hidden width (upstream 256), LayerNorm in the critic's hidden layers and an
  upper bound on TD targets are options; training statistics can be recorded.
- `replay_buffer.py`: batched insertion; an optional float32 tensor buffer kept on a device.
- `nn_utils.py`: `NeuralNet`, `pairwise_distances` and `weight_init`, copied unchanged.

`agent.py`, `env_wrapper.py` and `src/envs/pamdp/` are new code written for this repository.
With default options, `hyar_run.py` reproduces the original port bit for bit on CPU.
