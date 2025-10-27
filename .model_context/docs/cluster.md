# LRZ AI Systems - Complete Developer Documentation

> Comprehensive guide to the LRZ AI Systems cluster infrastructure for developers and researchers

---

## Table of Contents

1. [Access](#1-access)
2. [Compute Resources](#2-compute-resources)
3. [Storage](#3-storage)
4. [Enroot Containers](#4-enroot-containers)
5. [SLURM Job Management](#5-slurm-job-management)
   - [5.1 Interactive Jobs](#51-interactive-jobs)
   - [5.2 Single GPU Batch Jobs](#52-single-gpu-batch-jobs)
   - [5.3 Multi-GPU Batch Jobs](#53-multi-gpu-batch-jobs)
   - [5.4 Multi-Node Batch Jobs](#54-multi-node-batch-jobs)
6. [Interactive Apps](#6-interactive-apps)
7. [Datasets and Shared Containers](#7-datasets-and-shared-containers)
8. [Regulations and Policies](#8-regulations-and-policies)
9. [Support and Resources](#9-support-and-resources)

---

## 1. Access

### Requirements

- **LRZ User Account**: Valid LRZ account with AI-enabled project membership
- **Project Group**: Membership in `<projectID>-ai-c` group managed by Project Master User
- **Export Control**: Acceptance of export control statements via LRZ IDM portal

### Login Methods

**SSH Access:**
```bash
ssh <username>@login.ai.lrz.de
```

**Web Portal Access:**
- URL: [https://login.ai.lrz.de](https://login.ai.lrz.de)
- Provides both terminal and GUI-based access

### Initial Setup

1. Apply for LRZ account if you don't have one
2. Request AI project access through your Project Master User
3. Accept export control statements in IDM portal
4. Verify access by logging into `login.ai.lrz.de`
5. Check your home directory: `pwd` after login shows `/dss/dsshome1/<username>`

### Project Management

- **Master User**: Manages project membership, storage quotas, and resource allocation
- **User Types**: Regular users, Master Users, Data Curators
- **Access Control**: Centrally managed through LRZ systems

---

## 2. Compute Resources

### Available Partitions

| Partition Name            | Nodes | CPUs/Node | RAM/Node | GPUs/Node | GPU Model        | VRAM/GPU | Year |
|---------------------------|-------|-----------|----------|-----------|------------------|----------|------|
| lrz-hgx-h100-94x4         | 30    | 96        | 768 GB   | 4         | NVIDIA H100      | 94 GB    | 2022 |
| lrz-hgx-a100-80x4         | 5     | 96        | 1 TB     | 4         | NVIDIA A100      | 80 GB    | 2020 |
| lrz-dgx-a100-80x8         | 4     | 252       | 2 TB     | 8         | NVIDIA A100      | 80 GB    | 2020 |
| lrz-dgx-a100-40x8-mig     | 1     | 252       | 1 TB     | 8         | NVIDIA A100      | 40 GB    | 2020 |
| lrz-dgx-1-v100x8          | 1     | 76        | 512 GB   | 8         | NVIDIA V100      | 16 GB    | 2017 |
| lrz-dgx-1-p100x8          | 1     | 76        | 512 GB   | 8         | NVIDIA P100      | 16 GB    | 2016 |
| lrz-hpe-p100x4            | 1     | 28        | 256 GB   | 4         | NVIDIA P100      | 16 GB    | 2016 |
| lrz-v100x2 (default)      | 4     | 19        | 368 GB   | 2         | NVIDIA V100      | 16 GB    | 2017 |
| lrz-cpu                   | 12    | 18-94     | ≥360 GB  | -         | (CPU only)       | -        | -    |

### GPU Performance Comparison

| GPU Model | Architecture | FP32 TFLOPS | Tensor Cores FP16 TFLOPS | Memory (GB) |
|-----------|--------------|-------------|--------------------------|-------------|
| P100      | Pascal       | ~10         | -                        | 16          |
| V100      | Volta        | ~16         | ~125 (1st Gen)           | 16          |
| A100      | Ampere       | ~20         | ~312 (3rd Gen)           | 40/80       |
| H100      | Hopper       | ~51         | ~1000 (4th Gen)          | 94          |

### Partition Naming Convention

Format: `<housing>-<platform>-<GPU model>-<VRAM per GPU>x<number of GPUs>`

Example: `lrz-hgx-h100-94x4` = HGX housing, H100 GPU, 94GB VRAM, 4 GPUs per node

### Multi-Instance GPU (MIG)

- Available on A100 and H100 GPUs
- Partitions single GPU into smaller isolated instances
- Each instance has dedicated resources
- Useful for workloads that don't need full GPU

### Server Platforms

| Platform | Type                              | Built By       | Usage                           |
|----------|-----------------------------------|----------------|---------------------------------|
| HGX      | GPU baseboard & platform design   | OEMs + NVIDIA  | Used by partners to build servers |
| DGX      | Complete AI system                | NVIDIA         | Turnkey AI/HPC server           |

### Resource Selection Guidelines

- **CPU-only workloads**: Use `lrz-cpu` partition
- **Single GPU development/testing**: Use `lrz-v100x2` (default partition)
- **Large model training**: Use `lrz-hgx-h100-94x4` or `lrz-hgx-a100-80x4`
- **Multi-GPU distributed training**: Use DGX partitions with 8 GPUs
- **High-memory requirements**: Choose partitions with larger RAM per node

---

## 3. Storage

### Storage Tiers Overview

| Storage Type       | Mount Point         | Capacity         | Use Case                          | Backup              | Expiration          |
|--------------------|---------------------|------------------|-----------------------------------|---------------------|---------------------|
| Home Directory     | `/dss/dsshome1/`    | 100 GB           | Code, scripts, configs            | Yes (tape + snapshots) | Lifetime of project |
| AI Systems DSS     | `/dss/dssfs04`      | 4 TB (extendable)| High-bandwidth AI/ML workloads    | Optional (paid)     | Until further notice |
| Linux Cluster DSS  | `/dss/dssfs02-05/`  | 10 TB (extendable)| Long-term general storage        | Optional (paid)     | Lifetime of data project |
| Private DSS        | Custom paths        | Variable         | Dedicated private storage         | Owner-defined       | Owner-defined       |

### Home Directory

- **Path**: `/dss/dsshome1/<username>`
- **Capacity**: 100 GB per user
- **Purpose**: Store critical files like code, scripts, and configurations
- **Backup**: Automated backup to tape and file system snapshots
- **Performance**: Limited I/O bandwidth and latency
- **Access**: Automatically mounted at login (SSH and web)
- **Shared**: Unified home directory across LRZ AI Systems and Linux Cluster

**Check your location:**
```bash
pwd  # Shows /dss/dsshome1/<username> after login
```

### AI Systems DSS

- **Path**: `/dss/dssfs04`
- **Technology**: SSD-based network storage
- **Performance**: Optimized for high-bandwidth, low-latency I/O
- **Purpose**: High-intensity AI workloads and large-scale data operations
- **Quota**: Up to 4 TB, 8 million files, max 3 containers per project
- **Request**: Via LRZ Servicedesk by Project Master User
- **Expansion**: 4+ TB available with additional costs
- **Management**: Master User acts as DSS Data Curator

**Request form**: [https://servicedesk.lrz.de/en/ql/createsr/21](https://servicedesk.lrz.de/en/ql/createsr/21)

### Linux Cluster DSS

- **Path**: `/dss/dssfs02`, `/dss/dssfs03`, `/dss/dssfs05`
- **Capacity**: Up to 10 TB (20+ TB with costs)
- **Purpose**: General-purpose, long-term storage
- **Request Process**:
  1. Activate project for Linux Cluster: [Form](https://servicedesk.lrz.de/en/ql/createsr/12)
  2. Request explicit Linux Cluster DSS storage: [Form](https://servicedesk.lrz.de/ql/createsr/23)

### Private DSS

- **Deployment**: Dedicated DSS systems for private user groups
- **Purchase**: Part of joint project offering
- **Management**: System owner defines capacity, backup, and expiration policies
- **Examples**: `/dss/dsslegfs01`, `/dss/dsslegfs02`, `/dss/dssmcmlfs01`, `/dss/mcmlscratch`

### File System: IBM GPFS

**General Parallel File System (GPFS)** characteristics:

- **Architecture**: Distributed file system across multiple servers and disks
- **Parallel Access**: Thousands of compute nodes can access simultaneously
- **Metadata Management**: Distributed across servers (adds latency)
- **Inode Overhead**: Each file/directory requires an inode

### Storage Best Practices

**For Machine Learning Datasets:**

1. **Pack data into archives**: Use `.tar`, `.zip`, HDF5, or TFRecord formats
2. **Avoid small files**: Millions of individual images cause metadata bottleneck
3. **Sequential access**: Better than random reads/writes
4. **Minimize directory scans**: Avoid frequent `ls`, `find`, `stat` on large folders
5. **Use larger files**: Reduces inode usage and improves throughput

**Latency Considerations:**

- Creating/moving/deleting many small files is slower than on local disks
- Metadata operations distributed across servers add overhead
- Large numbers of files quickly reach inode limits and degrade performance

### Monitor Storage Usage

```bash
dssusrinfo all
```

Shows utilization overview of all accessible DSS containers.

### Additional Resources

- [File Systems and IO on Linux-Cluster](https://doku.lrz.de/file-systems-and-io-on-linux-cluster-10745972.html)
- [Data Science Storage Documentation](https://doku.lrz.de/data-science-storage-10745685.html)

---

## 4. Enroot Containers

### Overview

**Enroot** is NVIDIA's lightweight container framework for HPC environments:

- **Unprivileged**: No root access needed to build or run containers
- **User Space**: Operates entirely in user space
- **Format Support**: Docker images, NGC images, and custom builds
- **Integration**: Works with SLURM via Pyxis plugin
- **Isolation**: Separate file systems without heavy containerization overhead

### Key Characteristics

- **Not available on login nodes**: Use compute nodes via interactive sessions
- **Recommended approach**: Official method for custom software installation
- **No module/conda/pip**: These tools are not officially supported on AI Systems
- **Enroot-only support**: Use containers for all custom software needs

### Configuration

**Runtime configuration** via `~/.bashrc`:

```bash
# Disable writable root filesystem (containers read-only by default)
export ENROOT_ROOTFS_WRITABLE=no

# Disable automatic home directory mounting
export ENROOT_MOUNT_HOME=no
```

Or configure via `/etc/enroot/enroot.conf` (environment variables take precedence).

### Basic Workflow

#### Three Core Commands

1. **Import**: Download and convert image to Enroot format (`.sqsh`)
2. **Create**: Expand image into local container filesystem
3. **Start**: Run applications within the container

#### Import Container Image

```bash
# From Docker Hub
enroot import docker://ubuntu

# From NVIDIA NGC
enroot import docker://nvcr.io/nvidia/pytorch:23.10-py3

# Creates .sqsh file in current directory
```

#### Create Container

```bash
# Creates container from image (inherits image name without .sqsh)
enroot create ubuntu.sqsh

# Create with custom name
enroot create --name my_custom_name ubuntu.sqsh
```

#### Start Container

```bash
# Run interactive shell
enroot start ubuntu

# Execute specific command
enroot start ubuntu python3 train.py

# Run with root privileges (for installing software)
enroot start --root --rw ubuntu

# Inside container with root, install packages:
apt update && apt install -y build-essential cmake
exit
```

#### Export Modified Container

```bash
# Save changes to new .sqsh file
enroot export -o ubuntu-modified.sqsh ubuntu
```

#### List and Remove Containers

```bash
# List available containers
enroot list

# Remove container
enroot remove ubuntu
```

### NVIDIA NGC Integration

#### Register for NGC

1. Create NVIDIA profile: [https://ngc.nvidia.com/signin](https://ngc.nvidia.com/signin)
2. Browse containers in "Containers" tab
3. Search by keyword (e.g., PyTorch, TensorFlow)
4. Note image name and tag for import

#### Authenticate

Create `~/.enroot/.credentials` file:

```
machine nvcr.io login $oauthtoken password <YOUR_NGC_API_KEY>
machine authn.nvidia.com login $oauthtoken password <YOUR_NGC_API_KEY>

```

**Important**: Include newline at end of file.

**Get API Key:**
1. Sign in to NGC: [https://ngc.nvidia.com/catalog](https://ngc.nvidia.com/catalog)
2. Click user icon → Setup
3. Click "Get API Key" → "Generate API Key"
4. Save key securely

#### Import NGC Container

```bash
enroot import docker://nvcr.io/nvidia/pytorch:23.10-py3
```

**Important**: Verify container compatibility:
- Check NVIDIA driver version: `nvidia-smi` on compute node
- Verify CUDA version matches your needs
- Confirm preinstalled packages meet requirements

### Custom Image Creation

#### Extending Existing Images

**Example: Add Python packages to NGC PyTorch container**

```bash
# Start interactive session
salloc -p lrz-v100x2 --gres=gpu:1
srun --pty bash

# Import base image
enroot import docker://nvcr.io/nvidia/pytorch:23.10-py3

# Create container
enroot create --name my_pytorch pytorch.sqsh

# Start with write access and root
enroot start --root --rw my_pytorch

# Install packages inside container
pip3 install matplotlib seaborn wandb
apt update && apt install -y vim git

# Exit container
exit

# Export modified container
enroot export -o my_pytorch_custom.sqsh my_pytorch

# Clean up
enroot remove my_pytorch
exit  # Release allocation
```

#### For Non-NGC Images (Docker Hub, etc.)

Add CUDA environment variables to `/etc/environment` inside container:

```bash
enroot start --root --rw ubuntu

# Add to /etc/environment:
echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" >> /etc/environment
echo "NVIDIA_REQUIRE_CUDA=cuda>=9.0" >> /etc/environment
echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment

exit
```

These variables tell Enroot to inject NVIDIA libraries automatically.

### SLURM Integration (Pyxis Plugin)

#### Run Container in SLURM Job

```bash
# Interactive with container
srun --pty --container-image='nvcr.io#nvidia/pytorch:23.12-py3' bash

# Mount host directory into container
srun --pty \
  --container-mounts=./data:/mnt/data \
  --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
  bash

# Named container (persists within job allocation)
srun --pty \
  --container-name=my_session \
  --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
  bash
```

#### Batch Job with Container

```bash
#!/bin/bash
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -o log_%j.out

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     python3 train.py
```

### Container Best Practices

1. **Use NGC containers**: Optimized for NVIDIA GPUs, regularly updated
2. **Version control**: Tag your custom containers with versions
3. **Minimize layers**: Reduce container size and build time
4. **Document dependencies**: Note all installed packages for reproducibility
5. **Test compatibility**: Verify CUDA/driver versions before production runs
6. **Export after modifications**: Always export containers after installing software
7. **Clean up**: Remove unused containers with `enroot remove`

---

## 5. SLURM Job Management

### Overview

**SLURM (Simple Linux Utility for Resource Management)** is the workload manager for the LRZ AI Systems:

- **Resource allocation**: Manages CPUs, GPUs, memory, and time
- **Job scheduling**: Fair-share policies balance access among users
- **Queue management**: Jobs wait in queue if resources unavailable
- **Job types**: Interactive (real-time) and batch (automated)

### Essential SLURM Commands

```bash
# View available partitions and node status
sinfo

# View job queue (all users)
squeue

# View only your jobs
squeue --me

# Cancel a job
scancel <jobid>

# View detailed job information
scontrol show job <jobid>

# View partition details
scontrol show partition <partition_name>
```

---

## 5.1 Interactive Jobs

### Purpose

Interactive jobs provide direct access to compute resources for:
- Debugging and development
- Testing code before batch submission
- Installing software in containers
- Exploring datasets
- Short computational tasks

### Allocate Resources

```bash
# Request single GPU on default partition
salloc -p lrz-v100x2 --gres=gpu:1

# Request specific resources
salloc -p lrz-hgx-h100-94x4 --gres=gpu:2 --time=02:00:00

# Request CPU-only resources
salloc -p lrz-cpu --cpus-per-task=4 --mem=32G
```

**Common options:**
- `-p, --partition`: Partition name
- `--gres=gpu:N`: Request N GPUs (mandatory for GPU partitions)
- `--time`: Time limit (format: HH:MM:SS or D-HH:MM:SS)
- `--mem`: Memory per node (e.g., 32G, 64G)
- `--cpus-per-task`: CPU cores

### Launch Interactive Shell

After allocation, start shell on compute node:

```bash
# Basic bash shell
srun --pty bash

# With container
srun --pty --container-image='nvcr.io#nvidia/pytorch:23.12-py3' bash

# With mounted directories
srun --pty \
  --container-mounts=$HOME/data:/workspace/data \
  --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
  bash
```

### Inside Interactive Session

```bash
# Check GPU allocation
nvidia-smi

# Verify environment
hostname
env | grep SLURM

# Run your code
python3 train.py

# Exit when done (releases resources)
exit
```

### Exit Interactive Session

```bash
# Exit the interactive shell
exit

# If allocation still active, exit again to release
exit
```

---

## 5.2 Single GPU Batch Jobs

### Overview

Batch jobs are the **preferred method** for production workloads:
- Non-interactive execution
- Automatic queueing when resources unavailable
- Better resource utilization
- Reproducible job submission

### Basic Batch Script

Create `single_gpu_job.sbatch`:

```bash
#!/bin/bash
#SBATCH -p lrz-v100x2              # Partition name
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH -o log_%j.out              # Standard output file
#SBATCH -e log_%j.err              # Standard error file
#SBATCH --time=04:00:00            # Time limit (4 hours)
#SBATCH --job-name=my_training     # Job name

echo "=== Job started on $(hostname) at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"

# Load your environment or container
srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     python3 train.py --epochs 100 --batch-size 32

echo "=== Job finished at $(date) ==="
```

### Submit Job

```bash
sbatch single_gpu_job.sbatch
```

Output:
```
Submitted batch job 123456
```

### Advanced Batch Script

```bash
#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err
#SBATCH --time=08:00:00
#SBATCH --job-name=training_experiment_1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create log directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
nvidia-smi

# Run training with container
srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     --container-mounts=$HOME/data:/data,$HOME/outputs:/outputs \
     python3 train.py \
       --data /data/imagenet \
       --output /outputs/exp1 \
       --epochs 100 \
       --lr 0.001

echo "End time: $(date)"
```

### Common SBATCH Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `-p, --partition` | Partition name | `#SBATCH -p lrz-v100x2` |
| `--gres` | Generic resources (GPUs) | `#SBATCH --gres=gpu:1` |
| `-o` | Standard output file | `#SBATCH -o job_%j.out` |
| `-e` | Standard error file | `#SBATCH -e job_%j.err` |
| `--time` | Time limit | `#SBATCH --time=02:00:00` |
| `--job-name` | Job name | `#SBATCH --job-name=training` |
| `--mem` | Memory per node | `#SBATCH --mem=32G` |
| `--cpus-per-task` | CPU cores | `#SBATCH --cpus-per-task=4` |

### Monitor Jobs

```bash
# Check job status
squeue --me

# View job details
scontrol show job <jobid>

# View job output (while running)
tail -f log_<jobid>.out

# Cancel job if needed
scancel <jobid>
```

---

## 5.3 Multi-GPU Batch Jobs

### Overview

Multi-GPU jobs utilize multiple GPUs on a **single node** for:
- Data parallel training
- Model parallel training
- Large batch processing
- Multi-GPU inference

### Basic Multi-GPU Script

Create `multi_gpu_job.sbatch`:

```bash
#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4       # Partition with 4 GPUs
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err
#SBATCH --time=08:00:00
#SBATCH --job-name=multi_gpu_train

echo "Job started on $SLURM_NODELIST at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# PyTorch multi-GPU training
srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     python3 train_distributed.py \
       --nproc_per_node=4 \
       --epochs 100

echo "Job finished at $(date)"
```

### PyTorch Distributed Training

**Using `torch.distributed`:**

```bash
#!/bin/bash
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:8
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     python3 -m torch.distributed.launch \
       --nproc_per_node=8 \
       --nnodes=1 \
       --node_rank=0 \
       train.py --distributed
```

**Using `torchrun` (PyTorch 1.10+):**

```bash
srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     torchrun --nproc_per_node=8 train.py
```

### TensorFlow Multi-GPU

```bash
#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err

srun --container-image='nvcr.io#nvidia/tensorflow:23.10-tf2-py3' \
     python3 train_tf.py --num_gpus=4
```

### GPU Partition Selection

| Partition | GPUs/Node | Best For |
|-----------|-----------|----------|
| lrz-hgx-h100-94x4 | 4 | Latest GPU tech, large models |
| lrz-hgx-a100-80x4 | 4 | High-memory requirements |
| lrz-dgx-a100-80x8 | 8 | Maximum GPU count, largest models |
| lrz-dgx-1-v100x8 | 8 | Multi-GPU at scale |

### Best Practices

1. **Scale efficiently**: Test 1 GPU before scaling to multiple
2. **Monitor GPU utilization**: Use `nvidia-smi` to verify all GPUs are active
3. **Adjust batch size**: Larger batches with more GPUs
4. **Check memory**: Ensure model fits on all GPUs
5. **Use distributed frameworks**: PyTorch DDP, Horovod, DeepSpeed

---

## 5.4 Multi-Node Batch Jobs

### Overview

Multi-node jobs distribute workload across **multiple compute nodes** for:
- Very large model training (LLMs, foundation models)
- Distributed deep learning at scale
- High-throughput batch processing
- Parallel hyperparameter tuning

### Basic Multi-Node Script

Create `multi_node_job.sbatch`:

```bash
#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH -N 2                       # Request 2 nodes
#SBATCH --gres=gpu:4               # 4 GPUs per node
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err
#SBATCH --time=12:00:00
#SBATCH --job-name=multi_node_train

echo "Job started on nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 4))"

# Multi-node training command
srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     python3 -m torch.distributed.launch \
       --nproc_per_node=4 \
       --nnodes=$SLURM_JOB_NUM_NODES \
       --node_rank=$SLURM_NODEID \
       --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
       --master_port=29500 \
       train_multinode.py

echo "Job finished at $(date)"
```

### PyTorch Multi-Node Training

**Using `torchrun`:**

```bash
#!/bin/bash
#SBATCH -N 4
#SBATCH --gres=gpu:8
#SBATCH -p lrz-dgx-a100-80x8

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     torchrun \
       --nnodes=$SLURM_JOB_NUM_NODES \
       --nproc_per_node=8 \
       --rdzv_id=$SLURM_JOB_ID \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$(scontrol show hostname $SLURM_NODELIST | head -n1):29500 \
       train.py
```

### DeepSpeed Multi-Node

```bash
#!/bin/bash
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -p lrz-hgx-a100-80x4

# Create hostfile
scontrol show hostname $SLURM_NODELIST > hostfile

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     deepspeed --hostfile=hostfile \
       --num_nodes=$SLURM_JOB_NUM_NODES \
       --num_gpus=4 \
       train.py --deepspeed ds_config.json
```

### Horovod Multi-Node

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH -p lrz-hgx-a100-80x4

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' \
     horovodrun -np $((SLURM_JOB_NUM_NODES * 4)) \
       python3 train_horovod.py
```

### Multi-Node Configuration

**Key SLURM variables for multi-node:**

| Variable | Description |
|----------|-------------|
| `$SLURM_JOB_NUM_NODES` | Number of allocated nodes |
| `$SLURM_NODELIST` | List of allocated node names |
| `$SLURM_NODEID` | Node rank (0 to N-1) |
| `$SLURM_PROCID` | Process rank (global) |
| `$SLURM_LOCALID` | Process rank (per node) |

### Network and Communication

- **InfiniBand**: High-speed interconnect between nodes
- **NCCL**: NVIDIA Collective Communications Library for GPU communication
- **MPI**: Message Passing Interface for distributed computing
- **Master Node**: First node in allocation acts as coordinator

### Best Practices

1. **Start small**: Test on 1 node before scaling to multiple
2. **Network efficiency**: Use NCCL for GPU communication
3. **Load balancing**: Ensure data distributed evenly across nodes
4. **Fault tolerance**: Implement checkpointing for long jobs
5. **Monitor communication**: Check network utilization and bottlenecks
6. **Batch size scaling**: Increase batch size proportionally with node count

---

## 6. Interactive Apps

### Overview

LRZ AI Systems provide web-based interactive environments for development and analysis:

- **Access**: Browser-based at [https://login.ai.lrz.de](https://login.ai.lrz.de)
- **Containerized**: Each session runs in isolated container
- **Resource allocation**: Choose CPU/GPU, memory, and time
- **No installation**: Pre-configured environments ready to use

### Supported Applications

1. **Jupyter Notebook/Lab**: Python notebooks for data science and ML
2. **RStudio Server**: R environment for statistical computing
3. **TensorBoard**: Visualization for TensorFlow/PyTorch training

### Launch Procedure

#### Step 1: Select Application

- Login to [https://login.ai.lrz.de](https://login.ai.lrz.de)
- Navigate to **"Interactive Apps"** in top panel
- Choose application (e.g., "Jupyter Notebook")

#### Step 2: Configure Resources

**Resource Type:**
- CPU only
- CPU + single GPU

**Workload Specification:**
- CPU cores (e.g., 4, 8, 16)
- RAM requirements (e.g., 16GB, 32GB, 64GB)

**Container Environment:**
- Select specific version and pre-installed packages
- Each app provides multiple container options
- Containers include specific versions of Python/R and libraries

**Time Allocation:**
- Specify hours (e.g., 2h, 4h, 8h)
- **Warning**: Session will be terminated at time limit
- Unsaved work will be lost

#### Step 3: Launch Session

- Click **"Launch"** button
- Redirected to "My Interactive Sessions" page
- Session status progresses:
  - **Queued** → **Starting** → **Running**
- Typical startup time: < 2 minutes

#### Step 4: Connect

- Once status shows **"Running"**
- Button appears: **"Connect to [App Name]"**
- Click to open application in new browser tab
- Work in the environment as needed

#### Step 5: Manage Session

- **Monitor sessions**: "My Interactive Sessions" page shows all active sessions
- **Extend time**: Not possible; plan ahead
- **Save work frequently**: Auto-save may not cover all scenarios
- **End session early**: Delete session to free resources

### Jupyter Notebook/Lab

**Features:**
- Python 3.x with common ML libraries
- Pre-installed: NumPy, Pandas, Scikit-learn, Matplotlib
- GPU support with PyTorch/TensorFlow containers
- Terminal access for command-line operations

**Usage Tips:**
- Save notebooks regularly (`Ctrl+S`)
- Use `/dss/dssfs04` for large datasets
- Install additional packages: `!pip install <package>` (session-specific)
- Access GPU: `import torch; torch.cuda.is_available()`

### RStudio Server

**Features:**
- R environment with IDE
- Pre-installed packages for data analysis
- Version-specific containers available
- Package management within session

**Package Installation:**
- Session-specific: `install.packages("package_name")`
- Persistent: Use containers with pre-installed packages (see Section 6.1 R Package Management in Containers in LRZ docs)

### TensorBoard

**Features:**
- Visualize training metrics
- Compare multiple runs
- Examine model graphs
- Analyze performance

**Usage:**
- Point to log directory with training data
- Specify port (default: 6006)
- View metrics, graphs, and histograms

### Best Practices

1. **Save frequently**: Sessions end abruptly at time limit
2. **Use appropriate resources**: Don't request GPU if not needed
3. **Close when done**: Free resources for other users
4. **Store code in home**: Keep scripts in `/dss/dsshome1/`
5. **Use DSS for data**: Large datasets belong in `/dss/dssfs04`
6. **Test before batch**: Develop interactively, then submit batch jobs

### Limitations

- **Time limits**: Maximum session duration enforced
- **No session extension**: Cannot extend running session
- **No persistence**: Installed packages lost after session ends
- **Queue time**: May wait if resources are busy
- **Single user**: Sessions are not shared between users

---

## 7. Datasets and Shared Containers

### Problem Statement

AI/ML research often uses the same public datasets across multiple research groups:

- **Example**: AlphaFold database (>2TB) for protein structure prediction
- **Issue**: Multiple groups download identical data to personal storage
- **Result**: Data replication and wasted storage capacity

### Solution

**Dedicated DSS container** for centrally storing:
- Public datasets used by multiple researchers
- Enroot container images of common interest
- Reduces duplication and saves storage space

### Available Public Datasets

Centrally maintained datasets accessible to all users. Check current list via:
- LRZ documentation: [Available Public Datasets](https://doku.lrz.de/7-1-available-public-datasets-10747365.html)
- Or inquire via Service Desk

### Available Enroot Container Images

Pre-built containers for common workflows. Check current list via:
- LRZ documentation: [Available Enroot Container Images](https://doku.lrz.de/7-2-available-enroot-container-images-10747366.html)
- Or inquire via Service Desk

### Request New Public Dataset

#### Prerequisites

1. **Public license**: Dataset must be licensed for public usage
2. **No individual registration**: Must not require per-user licenses or registration
3. **Broad interest**: Justify usage by multiple researchers or groups

#### Request Process

1. **Open ticket** with LRZ Servicedesk: [https://servicedesk.lrz.de/en/ql/create/159](https://servicedesk.lrz.de/en/ql/create/159)
2. **Specify category**: 
   - "AI topics" → "Request new Dataset offer"
3. **Provide information**:
   - Dataset location (URL)
   - Justification for public interest
   - Expected target audience
   - Clear download instructions (ideally shell script)

#### Example Request

```
Category: AI topics → Request new Dataset offer

Description:
The AlphaFold dataset (https://alphafold.ebi.ac.uk/), which requires 
>2TB of storage, is becoming popular for protein prediction within the 
ML community. This dataset is used in methods X and Y.

The dataset is publicly available:
https://github.com/deepmind/alphafold#genetic-databases

Download instructions:
1. Install aria2c dependency
2. Execute: bash scripts/download_all_data.sh <DOWNLOAD_DIR>

Scripts available at:
https://github.com/deepmind/alphafold/tree/main/scripts/

Expected audience: Structural biology and computational biology research 
groups at LMU, TUM, and partner institutions.
```

### Request New Enroot Container Image

#### Prerequisites

1. **Public license**: Image must be publicly usable
2. **No individual registration**: Must not require per-user licenses
3. **Not in public registry**: Image not available on NVIDIA NGC, Docker Hub, or other public repositories
4. **Broad interest**: Justify usage by multiple researchers

#### Request Process

1. **Open ticket** with LRZ Servicedesk
2. **Provide**:
   - Location of Dockerfile
   - Justification for public interest
   - Expected target audience
   - Build instructions (if non-standard)

#### Example Request

```
Category: AI topics → Request new Container Image

Description:
Request to build and host a container image for the XYZ framework, which 
is commonly used in our research community but not available on NGC or 
Docker Hub.

Dockerfile location: https://github.com/org/xyz-framework/Dockerfile
Build instructions: Standard docker build (no special requirements)

Justification: This framework is used by 5+ research groups at LRZ 
partner institutions for quantum chemistry simulations. Pre-building 
would save time and ensure consistency.

Expected audience: Computational chemistry groups at TUM, LMU, MPG
```

### Important Notes

- **Acceptance subject to feasibility**: LRZ reviews all requests
- **Resource availability**: Limited by available storage and compute
- **Processing time**: Varies based on dataset size and complexity
- **Maintenance**: LRZ maintains centrally hosted datasets/images
- **Updates**: Dataset versions updated periodically; check documentation

### Benefits

1. **Storage efficiency**: No duplicate downloads
2. **Faster access**: Pre-downloaded and optimized
3. **Consistency**: Everyone uses same dataset version
4. **Maintenance**: LRZ handles updates and integrity checks
5. **Cost savings**: Reduced storage costs for users and LRZ

---

## 8. Regulations and Policies

### General Usage Regulations

- **Acceptance required**: Users must accept LRZ usage regulations
- **Export control**: Access subject to export control regulations
- **Fair use**: Resources shared among all authorized users
- **No commercial use**: Academic and research use only

### Data Privacy and GDPR

**GDPR Article 9 - Special Categories of Personal Data:**

- **Generally not permitted**: Processing of sensitive personal data restricted
- **Exceptions possible**: Require special agreement and secured infrastructure
- **Contact required**: Must request explicit approval via Service Desk

**If you need to process sensitive data:**
1. Open Service Desk ticket: [https://servicedesk.lrz.de/en](https://servicedesk.lrz.de/en)
2. Describe data type and processing requirements
3. Request special data processing agreement ("Auftragsverarbeitungsvertrag", AVV)
4. Wait for approval and infrastructure setup

### Project Data and Reporting

- **Electronic storage**: LRZ stores and may publish project data and descriptions
- **Waiver possible**: Can be requested explicitly
- **Reporting requirement**: Report on work and results may be requested by LRZ

### Acknowledgement in Publications

**Required acknowledgement template:**

```
This work was supported by the Leibniz Supercomputing Centre (LRZ) through 
access to the LRZ AI Systems / BayernKI infrastructure.
```

**Or use institution-specific template as provided by LRZ.**

### Export Control

- **Compliance required**: All users must comply with export control regulations
- **Acceptance in IDM**: Must accept statements in LRZ IDM portal
- **Restrictions apply**: Certain technologies and collaborations restricted

### Terms and Conditions

Full terms available at:
- [LRZ Usage Regulations](https://www.lrz.de/wir/einsicht/benutzungsrichtlinien/) (German)
- [Usage Regulations Translation](https://www.lrz.de/wir/einsicht/benutzungsrichtlinien_en/) (English, non-binding)

### Violations and Consequences

- **Suspension**: Access may be suspended for violations
- **Termination**: Severe violations may result in account termination
- **Legal action**: LRZ reserves right to pursue legal action if necessary

---

## 9. Support and Resources

### LRZ Service Desk

**Primary support channel:**
- URL: [https://servicedesk.lrz.de/en](https://servicedesk.lrz.de/en)
- Open tickets for:
  - Access issues
  - Storage quota requests
  - Dataset/container requests
  - Technical problems
  - Account management

### Documentation

**Main documentation hub:**
- [LRZ AI Systems Documentation](https://doku.lrz.de/ai-systems-11484278.html)

**Key documentation sections:**
- [1. Access](https://doku.lrz.de/1-access-10746642.html)
- [2. Compute](https://doku.lrz.de/2-compute-10746641.html)
- [3. Storage](https://doku.lrz.de/3-storage-10746646.html)
- [4. Enroot](https://doku.lrz.de/4-enroot-10746639.html)
- [5. Slurm](https://doku.lrz.de/5-slurm-1897076524.html)
- [6. Interactive Apps](https://doku.lrz.de/6-interactive-apps-10746644.html)
- [7. Datasets and Containers](https://doku.lrz.de/7-datasets-and-containers-10746647.html)

### Changelog

- **System updates**: Track infrastructure changes and updates
- **Maintenance**: Scheduled maintenance notifications
- **New features**: Announcements of new capabilities
- **URL**: [https://doku.lrz.de/9-changelog-10746645.html](https://doku.lrz.de/9-changelog-10746645.html)

### Training and Workshops

**AI Training Series:**
- Introduction to LRZ AI Systems
- Container technology for AI
- Distributed deep learning
- Performance optimization

Check LRZ website for upcoming training events.

### Community and Collaboration

**Partners and affiliations:**
- BayernKI initiative
- Munich universities (LMU, TUM)
- Max Planck Institutes
- Partner research institutions

### Quick Reference

**Login:**
```bash
ssh <username>@login.ai.lrz.de
# or visit https://login.ai.lrz.de
```

**Check resources:**
```bash
sinfo                    # Available partitions
squeue --me              # Your jobs
dssusrinfo all           # Storage usage
```

**Simple batch job:**
```bash
#!/bin/bash
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err

srun --container-image='nvcr.io#nvidia/pytorch:23.12-py3' python3 train.py
```

**Submit:**
```bash
sbatch job.sbatch
```

### Getting Help

1. **Check documentation first**: Most questions answered in docs
2. **Search changelog**: Recent changes may affect your workflow
3. **Open Service Desk ticket**: For unresolved issues
4. **Provide details**: Include job IDs, error messages, and steps to reproduce
5. **Be specific**: Clear problem descriptions get faster responses

---

## Appendix: Common Workflows

### Workflow 1: First-Time Setup

```bash
# 1. Login
ssh username@login.ai.lrz.de

# 2. Check home directory
pwd  # Should be /dss/dsshome1/username

# 3. Check storage
dssusrinfo all

# 4. Test interactive session
salloc -p lrz-v100x2 --gres=gpu:1
srun --pty bash
nvidia-smi
exit
exit
```

### Workflow 2: Custom Container Creation

```bash
# 1. Start interactive session
salloc -p lrz-v100x2 --gres=gpu:1
srun --pty bash

# 2. Import base image
enroot import docker://nvcr.io/nvidia/pytorch:23.12-py3

# 3. Create and modify container
enroot create --name my_env pytorch.sqsh
enroot start --root --rw my_env
pip install wandb optuna
exit

# 4. Export custom container
enroot export -o my_env.sqsh my_env

# 5. Clean up and exit
enroot remove my_env
exit
exit
```

### Workflow 3: Batch Job Submission

```bash
# 1. Prepare training script
vi train.py

# 2. Create batch script
vi job.sbatch

# 3. Submit job
sbatch job.sbatch

# 4. Monitor
squeue --me
tail -f log_<jobid>.out

# 5. Check results after completion
ls outputs/
```

### Workflow 4: Interactive Development

```bash
# 1. Login to web portal
# Visit https://login.ai.lrz.de

# 2. Select Interactive Apps → Jupyter Notebook

# 3. Configure: 1 GPU, 8 cores, 32GB RAM, 4 hours

# 4. Launch and wait for "Running" status

# 5. Click "Connect to Jupyter Notebook"

# 6. Develop and test code

# 7. Save final script to home directory

# 8. Submit as batch job for production run
```

---

## Document Information

**Version**: 1.0  
**Last Updated**: October 27, 2025  
**Source**: LRZ AI Systems Documentation  
**Maintained by**: LRZ Support Team

**Feedback**: Report documentation issues via [LRZ Service Desk](https://servicedesk.lrz.de/en)

---

**End of Documentation**
