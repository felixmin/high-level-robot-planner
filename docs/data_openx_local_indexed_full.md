# OpenX Local Indexed-Full Data

Current data path in this repo is:
- `data.backend=oxe_local_indexed`
- `data.adapter.openx_local.mode=indexed_full`
- local root: `data.adapter.openx_local.root` (default `/mnt/data/oxe`)

## How it works

1. Auto-discover dataset directories under `root` (if `auto_discover=true`).
2. Build or reuse episode index for train/val splits in:
   - `data.adapter.openx_local.index_cache_dir/episode_index/<key>`
3. Use indexed-full map dataset + hierarchical sampler at runtime.
4. Load only sampled episodes/timesteps during training.

## Primary knobs

- `data.adapter.openx_local.auto_discover`
- `data.adapter.openx_local.auto_train_split`
- `data.adapter.openx_local.auto_val_split`
- `data.adapter.openx_local.auto_pair_offset_steps`
- `data.adapter.openx_local.max_shards_per_dataset`
- `data.adapter.openx_local.pairs_per_episode`
- `data.adapter.openx_local.index_workers`
- `data.adapter.openx_local.index_rebuild`

## Typical commands

Stage 1 (LAQ):
```bash
python scripts/2_train_laq.py experiment=laq_oxe_local
```

Stage 2 (Foundation):
```bash
python scripts/4_train_foundation.py \
  experiment=vla_smol_flow_shared \
  model.laq.checkpoint=/path/to/laq.ckpt
```

## Notes

- No TF streaming backends are used in the active training path.
- If startup is slow, check whether index rebuild is enabled.
- Reuse index by keeping `index_rebuild=false`.
