# Documentation Index

Quick reference to all documentation in this project.

## Core Documentation

- **[../CLAUDE.md](../CLAUDE.md)** - Main project documentation for Claude Code
  - Project architecture, training stages, configuration system
  - Common workflows, dependencies, implementation notes

## Infrastructure & Cluster

- **[job_submission.md](job_submission.md)** - Job submission to LRZ cluster (⭐ Start here)
  - Single job and sweep submission with `submit_job.py`
  - CLI options, container config, monitoring
  - Design decisions and alternative approaches

- **[lrz_workflow.md](lrz_workflow.md)** - LRZ cluster setup and monitoring
  - Account setup, SSH config, storage quotas
  - Manual sbatch submission (legacy)

## Training Guides

- **[normal_training_guide.md](normal_training_guide.md)** - LAQ training reference
  - Multi-dataset configuration and filtering
  - Common commands and configuration examples
  - Troubleshooting guide

- **[profiling.md](profiling.md)** - Performance profiling
  - SimpleProfiler, AdvancedProfiler, PyTorchProfiler usage
  - Diagnosing slow training, common bottlenecks

## Data & Validation Systems

- **[validation_system.md](validation_system.md)** - Validation architecture
  - Bucket-strategy binding system
  - Validation strategies and metadata requirements

- **[dataset_adapter_plan.md](dataset_adapter_plan.md)** - Multi-dataset adapters
  - Unified adapter system for YouTube, BridgeV2 datasets
  - Configuration examples and metadata fields

- **[dataset_metadata.md](dataset_metadata.md)** - Dataset metadata reference
  - Metadata structure per dataset type
  - Filtering examples and access patterns

- **[oxe_datasets.md](oxe_datasets.md)** - OXE dataset integration
  - Open X-Embodiment dataset streaming
  - TensorFlow Datasets configuration

## Reference Documents

- **[oxe_bridge_memory.md](oxe_bridge_memory.md)** - Bridge dataset memory analysis
- **[lapa_comparison.md](lapa_comparison.md)** - LAPA project comparison
- **[validation_refactor_plan.md](validation_refactor_plan.md)** - Validation system design notes

---

## Quick Links

**Getting started:**
1. Read [../CLAUDE.md](../CLAUDE.md) for project overview
2. Submit jobs: [job_submission.md](job_submission.md)
3. Train locally: [normal_training_guide.md](normal_training_guide.md)

**Cluster submission:**
- [job_submission.md](job_submission.md) - Single jobs and sweeps

**Performance issues:**
- [profiling.md](profiling.md)

**Data loading:**
- [dataset_adapter_plan.md](dataset_adapter_plan.md) - Local datasets
- [oxe_datasets.md](oxe_datasets.md) - OXE streaming
