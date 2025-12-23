# Submitit Integration for LRZ Cluster

**Date:** 2025-12-23
**Status:** Partially working, needs container with dependencies

---

## Executive Summary

We attempted to integrate Hydra's submitit launcher for easier job submission to the LRZ cluster. The goal was to replace manual sbatch scripts with a single command like:

```bash
python scripts/submit_job.py experiment=laq_oxe_debug
```

**Current state:** The `submit_job.py` script works and submits jobs, but requires a container with all dependencies pre-installed.

---

## What We Installed

### 1. Dependencies Added to `environment.yml`

```yaml
- hydra-submitit-launcher>=1.2.0
- submitit>=1.5.0
```

These are only needed on the **login node** for job submission, not in the container.

### 2. Hydra Launcher Configs Created

```
config/hydra/launcher/
├── slurm.yaml           # Base Slurm settings
├── lrz_h100.yaml        # H100 profile with container support
└── lrz_h100_debug.yaml  # Debug profile (30 min, 1 GPU)
```

**Note:** These configs were created for the Hydra submitit-launcher plugin approach, which we later abandoned. They're still useful as reference but not currently used.

### 3. Scripts Created

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/submit_job.py` | Generate and submit sbatch scripts | ✅ Working |
| `scripts/enroot_wrapper.sh` | Wrapper for running inside container | ❌ Not used (abandoned approach) |

---

## Approaches Tried

### Approach 1: Hydra Submitit Launcher Plugin

**Idea:** Use `-m` flag with Hydra to automatically submit jobs:
```bash
python scripts/2_train_laq.py -m hydra/launcher=lrz_h100 experiment=laq_debug
```

**Why it failed:**
1. The training script imports `torch` at module level
2. Login node doesn't have torch installed
3. Script fails before Hydra even parses the config

```
File "scripts/2_train_laq.py", line 24, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**Would require:** Refactoring all training scripts to use lazy imports (imports inside `main()` function).

---

### Approach 2: Submitit with Pickle (Direct Executor)

**Idea:** Use submitit's `SlurmExecutor` directly to pickle a function and submit:

```python
executor = submitit.SlurmExecutor(folder="outputs/%j")
executor.update_parameters(
    partition="mcml-hgx-h100-94x4",
    srun_args=["--container-image=...", "--container-mounts=..."],
)
job = executor.submit(run_training, args)
```

**Why it failed:**
1. Submitit runs `python -m submitit.core._submit` on compute node
2. This requires `submitit` to be installed **inside the container**
3. Base PyTorch container doesn't have submitit

```
/usr/bin/python3: Error while finding module specification for
'submitit.core._submit' (ModuleNotFoundError: No module named 'submitit')
```

**Would require:** Installing submitit in the container, or building a custom container.

---

### Approach 3: Generate sbatch Scripts Directly (Current)

**Idea:** Generate sbatch scripts with container directives and submit via `sbatch` command:

```python
sbatch_content = f"""#!/bin/bash
#SBATCH --container-image={container}
#SBATCH --container-mounts={mounts}
...
python scripts/2_train_laq.py {overrides}
"""
subprocess.run(["sbatch", script_file])
```

**Status:** ✅ Works, but requires container with dependencies.

---

### Approach 4: Pip Install at Runtime

**Idea:** Install missing dependencies at job start:

```bash
pip install --target .pip-cache pytorch-lightning ...
export PYTHONPATH=.pip-cache:$PYTHONPATH
python scripts/2_train_laq.py ...
```

**Why it failed:**
1. Base container (nvidia pytorch:25.06) has PyTorch 2.5+
2. `pytorch-lightning` has version constraints that conflict
3. Pip cannot resolve dependencies

```
ERROR: ResolutionImpossible: pytorch-lightning versions have conflicting dependencies
```

**Would work if:** Using an older base container with compatible PyTorch version.

---

## Container Analysis

### Available Containers

| Container | PyTorch | Has Hydra | Has Lightning | Works |
|-----------|---------|-----------|---------------|-------|
| `nvidia+pytorch+25.06-py3.sqsh` | 2.5+ | ❌ | ❌ | ❌ Missing deps |
| `laq_pytorch_23.12_20251030_with_laq.sqsh` | 2.1 | ✅ | ❌ | ❌ Missing lightning |

### What's Needed in Container

For `scripts/2_train_laq.py` to run, the container needs:

```
- torch (with CUDA)
- pytorch-lightning / lightning
- hydra-core
- omegaconf
- transformers
- timm
- wandb
- einops
- webdataset
- accelerate
- tensorflow
- tensorflow-datasets
```

---

## Current Workflow

### What Works Now

1. **Submit jobs from login node:**
   ```bash
   python scripts/submit_job.py \
       --container /path/to/container/with/deps.sqsh \
       --time 01:00:00 \
       experiment=laq_oxe_debug
   ```

2. **Dry-run to see generated script:**
   ```bash
   python scripts/submit_job.py --dry-run experiment=laq_oxe_debug
   ```

3. **Override Slurm settings:**
   ```bash
   python scripts/submit_job.py \
       --partition mcml-hgx-h100-94x4 \
       --gpus 2 \
       --time 04:00:00 \
       --mem 128G \
       experiment=laq_full
   ```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--script`, `-s` | `2_train_laq` | Training script (without .py) |
| `--partition`, `-p` | `mcml-hgx-h100-94x4` | Slurm partition |
| `--gpus`, `-g` | `1` | Number of GPUs |
| `--time`, `-t` | `24:00:00` | Time limit |
| `--mem` | `64G` | Memory |
| `--cpus` | `8` | CPUs per task |
| `--container` | env or default | Container image path |
| `--pip-install` | false | Install deps at runtime (broken) |
| `--dry-run` | false | Print script without submitting |

---

## What's Missing

### 1. Container with All Dependencies

**Priority: HIGH**

Need to build a container with all required packages. Options:

a) **Extend existing container:**
   ```bash
   enroot start --root --rw laq_container
   pip install pytorch-lightning transformers ...
   enroot export -o hlrp_container.sqsh laq_container
   ```

b) **Build from Dockerfile:**
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:23.12-py3
   RUN pip install pytorch-lightning hydra-core ...
   ```

### 2. Set Default Container Path

Once container is built, update default in `submit_job.py`:

```python
container_image = args.container or os.environ.get(
    "HLRP_CONTAINER_IMAGE",
    "/dss/dsshome1/00/go98qik2/workspace/containers/hlrp.sqsh"  # New container
)
```

### 3. Sweep Support (Future)

Currently single jobs only. For sweeps, could add:

```bash
python scripts/submit_job.py \
    experiment=laq_full \
    'training.optimizer.lr=1e-4,5e-5,1e-5'  # Submit 3 jobs
```

### 4. Documentation Updates

- Update `CLAUDE.md` with new workflow
- Update `docs/lrz_workflow.md`

---

## Files Changed

| File | Change |
|------|--------|
| `environment.yml` | Added hydra-submitit-launcher, submitit |
| `config/hydra/launcher/*.yaml` | Created (not currently used) |
| `scripts/submit_job.py` | Created - main submission script |
| `scripts/enroot_wrapper.sh` | Created (not currently used) |
| `slurm/train_language_table.sbatch` | Updated with container directives |

---

## Recommendations

### Short-term (Get it Working)

1. Build container with all dependencies
2. Test with: `python scripts/submit_job.py --container /path/to/new.sqsh experiment=laq_oxe_debug`
3. Set as default container in script

### Long-term (Production Ready)

1. Add sweep support to `submit_job.py`
2. Consider refactoring training scripts for lazy imports (enables Hydra launcher plugin)
3. Add job monitoring/status commands
4. Integrate with wandb sweeps

---

## Commands Reference

```bash
# Submit single job
python scripts/submit_job.py experiment=laq_oxe_debug

# With custom container
python scripts/submit_job.py --container /path/to/container.sqsh experiment=laq_oxe_debug

# Dry run
python scripts/submit_job.py --dry-run experiment=laq_oxe_debug

# Custom resources
python scripts/submit_job.py --gpus 4 --time 08:00:00 experiment=laq_full

# Monitor jobs
squeue --me
tail -f outputs/logs/<job_id>.out
scancel <job_id>
```
