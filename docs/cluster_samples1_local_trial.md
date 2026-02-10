# Cluster Trial Plan: `samples_per_episode=1`

## Situation
- `samples_per_episode=1` is currently much slower than all-pairs in local GCS streaming tests.
- Two major confounders in local tests:
  - remote GCS network latency/variance,
  - per-dataset cold-start/switch overhead on first reads.
- On cluster, data is expected to be available in a local TFDS mirror, so this mode may be significantly faster there.

## Config To Use
- Primary config (local mirror + one sample per episode):
  - `data=laq_oxe_cluster_mirror_large_local_python_hot_samples1`
  - file: `config/data/laq_oxe_cluster_mirror_large_local_python_hot_samples1.yaml`
- If mirror root differs, override:
  - `data.adapter.tf.tfds_read.local_root=<cluster-openx-root>`

## What To Try On Cluster
Run short throughput sweeps first (same model, small `max_steps`), then one longer confirmation run.

1. Baseline (samples1, local mirror)
   - keep config defaults
   - measure batches/s and samples/s after warmup

2. Mixing aggressiveness sweep
   - `data.adapter.tf.mixing.mix_block_length={2,4,8}`
   - goal: reduce switch overhead without degrading mixing too much

3. Python prefetch depth sweep
   - `data.adapter.tf.mixing.python_prefetch_queue_size={2,4,8}`
   - goal: check if deeper per-dataset ready queue helps samples1 mode

4. Read parallelism sweep
   - `data.adapter.tf.tfds_read.decode_parallelism={4,8}`
   - `data.adapter.tf.tfds_read.interleave_parallelism={4,8}`
   - goal: verify local storage can exploit higher parallel reads

## Success Criteria
- Minimum target: >= 250 samples/s at effective batch size 128 equivalent.
- Preferred target: >= 320 samples/s (buffer for validation / occasional stalls).
- Also verify no dataset-specific failures in priming logs.

## Notes
- Keep `return_metadata=false` for these throughput runs (`pair_256_no_meta`) to measure the training hot path.
- If results are good on cluster-local reads, keep this config as the default for samples1 experiments and use all-pairs only when explicitly needed.
