# Documentation Index

Quick reference to all documentation in this project.

## Core Documentation

- **[../CLAUDE.md](../CLAUDE.md)** - Main project documentation for Claude Code
  - Project architecture, training stages, configuration system
  - Common workflows, dependencies, implementation notes

## Training Guides

- **[normal_training_guide.md](normal_training_guide.md)** - Quick reference for normal LAQ training
  - Data loading modes (scene-level vs pair-level vs debug)
  - Common commands and configuration examples
  - Troubleshooting guide

- **[profiling.md](profiling.md)** - Performance profiling guide
  - SimpleProfiler, AdvancedProfiler, PyTorchProfiler usage
  - Diagnosing slow training, common bottlenecks

## Infrastructure

- **[lrz_workflow.md](lrz_workflow.md)** - LRZ cluster workflow
  - Setup, job submission, monitoring on H100 cluster

## Implementation Plans (Future Work)

- **[dataset_adapter_plan.md](dataset_adapter_plan.md)** - Multi-dataset adapter architecture
  - Plan for unifying YouTube, BridgeV2, OpenX datasets
  - Adapter pattern implementation details

- **[dataset_metadata.md](dataset_metadata.md)** - Dataset metadata reference
  - Metadata structure per dataset type
  - Filtering examples and access patterns

---

## Quick Links

**Getting started:**
1. Read [../CLAUDE.md](../CLAUDE.md) for project overview
2. Follow [normal_training_guide.md](normal_training_guide.md) for training

**Performance issues:**
- See [profiling.md](profiling.md)

**LRZ cluster:**
- See [lrz_workflow.md](lrz_workflow.md)
