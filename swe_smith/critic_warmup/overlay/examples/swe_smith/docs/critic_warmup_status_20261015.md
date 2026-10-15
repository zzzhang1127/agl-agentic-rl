# MiniCPM5 critic warmup: engineering acceptance, 2026-10-15

This is bounded PPO/GAE critic warmup with a fixed actor, not an accepted full
SAO reproduction. Actor updates remain disabled. All 42 critic attention layers
are frozen; the remaining supported critic parameters train.

The validated two-GPU configuration uses batch 2, learning rate 5e-6, one
critic epoch, a total budget of 200 steps and checkpoint saves every 20 steps
(roughly half an hour). vLLM uses TP=1, max_num_seqs=1, context 13312,
gpu_memory_utilization=0.35 and 1024 explicit KV blocks. Long-context generation
was checked at 11264 input plus 2048 output tokens. Observed vLLM peak was
9538 MiB per GPU; the whole experiment peaked around 25 GiB per GPU.

Training preserves the complete merged trajectory: prompt and response-span
caps are both 13312, with worker token capacity 26624. The merged span includes
tool output and can exceed the generated-token budget. Sampled token IDs and
behavior log probabilities must align; infrastructure errors and missing rewards
must be rejected rather than converted to zero reward.

Critic-only native resume was verified on real GPUs with model, optimizer,
RNG/scheduler and dataloader state, followed by an actual update and save.
Atomic saving checks every rank's model/optimizer/extra shard and data.pt,
publishes the directory, atomically updates the tracker, then removes the old
checkpoint. A temporary second copy during save is necessary for crash safety.

The last run stopped on disk protection after reaching step 59; the latest
complete checkpoint is step 40. Unsaved steps 41–59 are not recoverable. Reward
and token alignment checks passed, but explained variance remained negative.
The diagnostic dataset has only 32 training tasks and one validation task.
Neither decreasing loss nor these training batches establishes value quality.
Independent positive/negative held-out trajectories, comparison against constant
and initialized predictors, stable explained variance and calibration are needed
before considering actor training. The evaluation set must not select training tasks.

## Storage policy

All experiment weights, archives, logs, Ray temporary files and compilation
caches belong on the data filesystem. Do not fall back to the root filesystem
when data is full. `examples/swe_smith/critic_storage.py` provides checked runtime
environment paths; apply its returned dictionary to `os.environ` before importing
training libraries. Set checkpoint output explicitly under the same project.
For example, `runtime_environment('/data/user/critic', run_name)` produces short
Ray paths and redirects caches and archives. Existing symlink escapes are rejected.
Keep enough data space for an entire additional checkpoint and concurrent growth;
this experiment's complete checkpoint is approximately 22.2 GiB. Its data guard
is 32 GiB and restart admission is 64 GiB. No automatic actor activation is allowed.

`python examples/swe_smith/run_critic_lowmem.py --help` shows the portable
two-GPU launcher. It requires explicit model/data paths, the deployed Kubernetes
job template, a compatible MiniCPM tool-parser plugin and chat template, and a
private AGL key file. Supply `SWANLAB_API_KEY` through the environment. It
applies the low-memory overrides after the legacy smoke profile, rejects GPU5
and root storage, and validates an explicit resume checkpoint. Resuming requires
the same checkpoint output root to avoid retaining a second copy. The launcher
itself is not a disk supervisor: run it under a supervisor which checks data
headroom, finite metrics, actor interlocks and reward supply throughout training.
