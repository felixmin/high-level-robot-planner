A Technical Blueprint for Scalable Implementation of Latent Action Pretraining from Videos on HPC InfrastructureExecutive SummaryThis document presents a comprehensive technical blueprint for the implementation of the Latent Action Pretraining from Videos (LAPA) project. The proposed architecture is designed for scalability, reproducibility, and efficient utilization of the Leibniz Supercomputing Centre (LRZ) High-Performance Computing (HPC) cluster. The design addresses the three core stages of the LAPA system: Latent Action Quantization (LAQ), the Foundation Policy, and the Low-Level Policy. It moves beyond a proof-of-concept to establish a robust, production-ready research platform capable of handling petascale datasets and training state-of-the-art Vision-Language-Action models.The core strategy is built upon a hybrid approach, selecting the most appropriate tools and frameworks for each component's specific needs. A modular monorepo is recommended for unified version control and atomic commits across the interdependent project stages. Data I/O, a critical bottleneck in HPC environments, is addressed by adopting the sharded WebDataset format, which circumvents the performance limitations of GPFS filesystems with large numbers of small files. For training, a dual-framework strategy is proposed: PyTorch Lightning for the relatively standard LAQ and Low-Level policy stages to accelerate experimentation, and a combination of raw PyTorch with Lightning Fabric for the large-scale Foundation Policy to retain maximum control over the advanced distributed training loop. The entire system is orchestrated by Hydra for flexible and reproducible configuration management.For deployment on the LRZ cluster, this blueprint details a complete workflow using the mandated Enroot containerization system, integrated with the Slurm workload manager via the Pyxis plugin. Special attention is given to scaling the Foundation Policy across multiple H100 nodes using Fully Sharded Data Parallel (FSDP), with specific recommendations for performance optimization, including mixed-precision training and efficient checkpointing strategies. The following table summarizes the key architectural decisions that form the foundation of this report.Architectural ComponentRecommendationPrimary JustificationRepository StructureModular MonorepoMaximizes code reuse and enables atomic commits across interdependent project stages, ensuring system-wide consistency.Data I/O FormatSharded WebDataset (TAR archives)Overcomes the GPFS filesystem bottleneck with small files by enabling high-throughput, sequential I/O from large archives.Configuration ManagementHydraProvides a powerful, compositional framework for managing complex experimental configurations and command-line overrides.Training Framework (LAQ/Low-Level)PyTorch LightningReduces boilerplate code and enforces a clean structure for standard supervised training loops, accelerating experimentation.Training Framework (Foundation)Raw PyTorch + Lightning FabricOffers maximum control and flexibility for the complex, large-scale VLM training loop while abstracting away hardware orchestration.Distributed Training StrategyFully Sharded Data Parallel (FSDP)Enables training of models that exceed single-GPU memory by sharding parameters, gradients, and optimizer states across devices.HPC ContainerizationEnrootAdheres to LRZ cluster requirements, providing a reproducible and isolated environment for training and inference.HPC Job Submissionsbatch Templates + PyxisProvides a robust and transparent method for launching containerized, multi-node jobs on Slurm, integrated with Hydra for configuration.This document serves as a definitive guide for building a scalable and maintainable system for the LAPA project, providing detailed architectural plans, component interface designs, and HPC-specific deployment workflows.1. A Unified Foundation: The Modular Monorepo ArchitectureThe structural organization of a research project's codebase is a foundational decision that profoundly impacts development velocity, collaboration, and long-term maintainability. For a multi-stage, deeply interconnected project like LAPA, a Modular Monorepo architecture is the superior choice over alternatives like a multi-repository (microservices) approach.1.1. Rationale for a Monorepo in a Multi-Stage Research ProjectThe LAPA system is not a collection of independent services but a sequential pipeline where the three stages—LAQ, Foundation, and Low-Level—are tightly coupled. The latent action space defined and produced by the LAQ model serves as the predictive target for the Foundation Policy. In turn, the discrete latent codes predicted by the Foundation Policy are the primary input for the Low-Level Policy. This tight coupling means that a change in one component frequently necessitates corresponding changes in others.For example, modifying the dimensionality or vocabulary size of the latent action space in the LAQ model requires immediate and corresponding adjustments to the output head of the Foundation Policy and the input layer of the Low-Level Policy. In a multi-repository architecture, this single logical change would necessitate three separate pull requests across three different repositories, each requiring careful version management and coordinated deployment to avoid integration failures. This introduces significant friction and a high risk of version-skew errors, where incompatible versions of the components are used together, leading to subtle bugs that are difficult to diagnose.A monorepo resolves these challenges by collocating all project code in a single repository. This enables atomic commits that span multiple components. A single commit can simultaneously update the LAQ model, the Foundation Policy, the Low-Level Policy, and any shared configuration files, guaranteeing that the entire system is always in a consistent and valid state. This property is not merely a convenience; it is a critical mechanism for ensuring scientific rigor and reproducibility. It eliminates an entire class of errors related to component version mismatch and dramatically simplifies the process of checking out a previous version of the project to reproduce a specific result. Furthermore, it simplifies dependency management and streamlines the setup of continuous integration (CI) and automated testing pipelines. While frameworks like PyTorch Lightning enforce structure at the code level 1, the monorepo extends this principle to the entire system architecture.1.2. Proposed Directory Structure and Component BoundariesTo balance the benefits of a unified repository with the need for clear component separation and modularity, the following directory structure is proposed:Plaintextlapa-project/
├── containers/
│   └── Dockerfile             # Dockerfile for building the Enroot container
├── config/
│   ├── experiment/            # Complete experiment configs (e.g., full_pipeline_v1.yaml)
│   ├── model/                 # Model-specific configs (laq.yaml, foundation.yaml)
│   ├── data/                  # Dataset configs (openx.yaml)
│   ├── training/              # Training loop configs (optimizer.yaml, scheduler.yaml)
│   └── hydra/                 # Hydra-specific configs (launcher, sweeper)
├── packages/
│   ├── common/                # Shared utilities, interfaces, and constants
│   │   ├── setup.py
│   │   └── src/common/...
│   ├── laq/                   # LAQ model, task, and data logic
│   │   ├── setup.py
│   │   └── src/laq/...
│   ├── foundation/            # Foundation Policy model and training logic
│   │   ├── setup.py
│   │   └── src/foundation/...
│   └── low_level/             # Low-Level Policy model and training logic
│       ├── setup.py
│       └── src/low_level/...
├── scripts/
│   ├── preprocess_data.py     # Script to create WebDataset shards
│   ├── train.py               # Unified training entry point
│   └── evaluate.py            # Evaluation script
├── slurm/
│   └── train.sbatch           # Template Slurm batch script for LRZ
└── pyproject.toml             # Project metadata and top-level dependencies
This structure provides several key advantages:Installable Packages: Each component under packages/ is a self-contained Python package with its own setup.py. This allows a developer to work on a single component in isolation by installing it in editable mode (e.g., pip install -e packages/laq). This is crucial for unit testing and focused development.Centralized Configuration: All Hydra configuration files are located in a single config/ directory. This promotes consistency and enables the creation of unified experiment configurations that define parameters for the entire pipeline. For instance, an experiment file can specify which LAQ checkpoint to use when training the Foundation Policy, ensuring a clear and version-controlled link between the stages.Clear Separation of Concerns: The structure cleanly separates the core Python logic (packages/), configuration (config/), operational scripts (scripts/), container definitions (containers/), and HPC job templates (slurm/). This clarity makes the project easier to navigate and maintain.1.3. Dependency Management and Environment IsolationTo prevent the dependency conflicts that can sometimes arise in large monorepos, a layered dependency management strategy will be employed.Core Dependencies: Common, shared dependencies (e.g., torch, pytorch-lightning, hydra-core) will be defined in the top-level pyproject.toml file. These form the base environment for the entire project.Component-Specific Dependencies: Each package in packages/ can specify any additional, unique dependencies in its own setup.py file. For example, if the foundation package requires the transformers library but the other packages do not, it will be listed only in packages/foundation/setup.py.This approach ensures that the shared environment is kept lean, while allowing individual components the flexibility to include specialized libraries without polluting the global namespace or creating conflicts. The final, unified environment used for training will be built into the Enroot container, capturing the complete set of dependencies required by all components.2. Data Preprocessing and I/O Strategy for Petascale DatasetsThe performance of any large-scale deep learning system is fundamentally constrained by its ability to feed data to the GPUs. On HPC clusters like LRZ, which utilize parallel file systems, the data format and I/O strategy are not minor implementation details but critical architectural decisions that can determine the success or failure of the project.2.1. The GPFS Bottleneck: Why File-per-Frame Fails at ScaleThe LRZ cluster's storage, including the high-performance /dss/dssfs04 filesystem, is based on the General Parallel File System (GPFS), also known as IBM Spectrum Scale. GPFS is designed for high-bandwidth, streaming I/O on very large files. Its performance excels when reading or writing gigabytes of data sequentially. However, it performs poorly when subjected to workloads that require high rates of random access to millions of small files, a pattern known as high IOPS (Input/Output Operations Per Second).A robotics video dataset can easily contain tens of millions of individual frames. Storing each frame as a separate file (e.g., .jpg or .png) creates a catastrophic I/O pattern for GPFS. Every file open, read, and close operation requires interaction with the filesystem's metadata servers. When thousands of processes across hundreds of nodes attempt to access millions of tiny files simultaneously, the metadata servers become overwhelmed. This leads to extreme latency, with I/O speeds plummeting to a fraction of the hardware's theoretical capability. The result is a data loading pipeline that cannot keep the GPUs saturated, causing them to sit idle for most of the training cycle and wasting valuable computational resources. This is a well-documented anti-pattern in HPC, and it is imperative to avoid it.2.2. The WebDataset Solution: Sequential I/O with TAR ArchivesTo overcome the GPFS bottleneck, the recommended solution is to adopt the WebDataset format for all training data.3 WebDataset is a PyTorch I/O library designed specifically for large-scale training. It addresses the small-file problem by packing large numbers of training samples into sharded, sequential archives, typically in the standard POSIX TAR format.4The core principles of this approach are:Sharding: The total dataset is split into numerous large files called shards (e.g., 1024 shards of 1-2 GB each).Sequential Access: During training, each worker process reads from a specific set of shards. It reads each TAR file sequentially from beginning to end, which is the optimal access pattern for GPFS and other parallel file systems. This maximizes I/O bandwidth.Local Caching/Buffering: Data is read from the TAR archives in large chunks, and samples are buffered in memory for shuffling and processing.This strategy effectively transforms a high-IOPS, random-access problem into a low-IOPS, high-bandwidth streaming problem, aligning the I/O pattern with the strengths of the underlying HPC storage system. Benchmarks have shown that this can lead to a tenfold improvement in I/O performance compared to file-per-sample approaches.4 The use of a standard format like TAR also simplifies data distribution and management, as the datasets can be used directly without needing to be unpacked.52.3. Low-Level Design: TAR Archive Structure and Preprocessing WorkflowA one-time, offline preprocessing step is required to convert the raw video datasets into the sharded WebDataset format. This will be accomplished by a parallelized Python script located at scripts/preprocess_data.py. This script will read the source data and write the resulting TAR shards to the /dss/dssfs04 high-performance storage area.Inside each TAR shard, a strict naming convention must be followed, as this is how WebDataset groups files into a single training sample.4 All files belonging to one sample must share the same basename. For a given sample with the unique key episode_00123_step_045, the TAR archive would contain files such as:episode_00123_step_045.0.jpg: The first frame of the frame-pair.episode_00123_step_045.1.jpg: The second frame of the frame-pair.episode_00123_step_045.json: A JSON file containing metadata, such as the natural language instruction associated with the episode (e.g., {"instruction": "pick up the red block"}).episode_00123_step_045.proprio.pyd: A file containing proprioceptive state information, if available.WebDataset automatically groups all files with the episode_00123_step_045 prefix into a single dictionary-like sample object. This flexible format can easily accommodate the different data requirements of the three LAPA stages.2.4. Implementation: The LAPADataModuleData loading for all training stages will be encapsulated within a LAPADataModule class, which inherits from pytorch_lightning.LightningDataModule. This class will provide a standardized interface for accessing training, validation, and test data.The core of the LAPADataModule will be the instantiation and configuration of a webdataset.DataPipeline. This pipeline defines the sequence of operations applied to the data stream.Interface Definition: LAPADataModulePlaintextclass LAPADataModule(pl.LightningDataModule):
    def __init__(self, shard_urls: list[str], batch_size: int, num_workers: int):
        #... initialization...

    def setup(self, stage: str | None = None):
        # This method is called on every GPU process
        self.dataset = self._build_pipeline()

    def _build_pipeline(self) -> wds.DataPipeline:
        # Construct the WebDataset processing pipeline
        # 1. Start with the list of TAR shard URLs
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.shard_urls),
            # 2. Distribute shards across nodes and workers for DDP/FSDP
            wds.split_by_node,
            wds.split_by_worker,
            # 3. Shuffle the order of shards for this worker
            wds.shuffle(1000),
            # 4. Decode the TAR file stream and yield samples
            wds.tarfile_to_samples(),
            # 5. Shuffle samples within a rolling buffer
            wds.shuffle(10000),
            # 6. Decode specific file types (e.g., jpg to tensors)
            wds.decode("torchrgb"),
            # 7. Map sample dict to a tuple for model input
            wds.to_tuple("0.jpg", "1.jpg", "json"),
            # 8. Batch the samples
            wds.batched(self.batch_size, partial=False)
        )
        return pipeline

    def train_dataloader(self) -> DataLoader:
        # batch_size=None because batching is done inside the dataset pipeline
        return DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers)

    #... val_dataloader() and test_dataloader()...
A crucial aspect of this implementation is the correct handling of distributed sampling. The inclusion of wds.split_by_node and wds.split_by_worker in the pipeline is essential.6 These stages ensure that when training with multiple GPUs across multiple nodes, each worker process is assigned a unique, non-overlapping subset of the data shards. Without this, each worker would process the entire dataset, leading to incorrect training and skewed evaluation results.Furthermore, this streaming data approach necessitates a subtle but important shift in the concept of a training epoch. With a traditional DataLoader, shuffling is deterministic based on a seed. With WebDataset, shuffling is approximate, based on shard order and a rolling buffer.8 This means that an "epoch" is no longer a pass over the data in a fixed, albeit shuffled, order. The more robust and scalable paradigm, common in large-scale training, is to define an epoch as a fixed number of training steps (e.g., 50,000 gradient updates).9 This makes training progress independent of the stochasticity of the data pipeline and simplifies checkpointing and resumption.3. A Hybrid Training Framework for Research and ScalabilityThe choice of a training framework involves a fundamental trade-off between abstraction and control. A high-level framework can accelerate development by removing boilerplate code, while a lower-level approach provides the fine-grained control necessary for implementing cutting-edge or complex training algorithms. Given the varying complexity of the LAPA project's stages, a single, monolithic framework choice is suboptimal. Instead, a hybrid strategy that applies the principle of appropriate abstraction is recommended.3.1. The Principle of Appropriate AbstractionThe three training stages of LAPA have distinct characteristics:LAQ and Low-Level Policy: These are likely to be standard supervised learning problems. The LAQ model is trained with a reconstruction and commitment loss, and the Low-Level Policy is trained to regress from latent actions to robot commands. Their training loops are conventional.Foundation Policy: This is the core research component—a large-scale Vision-Language-Action model. Its training will involve advanced techniques such as multi-node distributed training with FSDP, custom activation checkpointing strategies, and potentially complex sharding and checkpointing logic that requires deep control over the training process.Therefore, the framework choice should be tailored to the task. Using a high-level abstraction where it fits (LAQ, Low-Level) will save development time, while using a more flexible, lower-level approach where needed (Foundation) will enable advanced research without fighting the framework.3.2. Recommendation for LAQ & Low-Level Policy: PyTorch LightningFor the LAQ and Low-Level Policy training stages, PyTorch Lightning is the recommended framework. PyTorch Lightning is a lightweight wrapper on top of PyTorch that structures the code and automates the engineering aspects of training.1 By encapsulating the model logic in a LightningModule and the data logic in a LightningDataModule, it removes the need to write manual training and validation loops, optimizer steps, backpropagation calls, and device placement logic.1This provides several key benefits for these stages:Reduced Boilerplate: Developers can focus on the research-specific components—the model architecture (__init__), the forward pass (training_step), and the optimizer configuration (configure_optimizers)—rather than rewriting repetitive training loop code.2Best Practices by Default: Lightning's Trainer object transparently handles details like checkpointing, logging to services like Weights & Biases, multi-GPU training with DDP, and mixed-precision training, all through simple configuration flags.1Reproducibility: The structured nature of Lightning code makes it more readable, maintainable, and easier for new team members to understand, which is crucial for reproducible research in a team setting.2While there can be a minor performance overhead compared to a highly optimized raw PyTorch loop, for standard training tasks this is typically negligible and far outweighed by the significant increase in development velocity and reduction in potential bugs.103.3. Recommendation for Foundation Policy: Raw PyTorch + Lightning FabricFor the complex, large-scale training of the Foundation Policy, a more flexible approach is necessary. The recommendation is to use a raw PyTorch training loop augmented with Lightning Fabric.Lightning Fabric is designed for use cases where full control over the training loop is essential, but the complexities of orchestrating distributed training are still undesirable.12 It provides the "scaffolding" for multi-node, multi-device training without imposing the rigid structure of the full Lightning Trainer.13This hybrid approach offers the best of both worlds:Full Control: The training loop remains pure PyTorch. This allows for the implementation of custom logic that might be difficult or impossible with a more abstracted Trainer, such as fine-grained control over FSDP wrapping policies, explicit communication prefetching, or non-standard optimizer steps. This agility is critical for a component that is the subject of active, state-of-the-art research.Hardware Abstraction: Fabric handles the complex and error-prone aspects of setting up distributed process groups, managing device placement, and wrapping the model and optimizer for distributed strategies like FSDP. The developer simply calls fabric.setup(model, optimizer) and fabric.backward(loss), and Fabric ensures the correct distributed operations occur under the hood.12This choice future-proofs the core research component of the project. As new techniques for large model training emerge, they can be directly implemented in the raw PyTorch loop, ensuring that the infrastructure can adapt to new scientific discoveries without requiring a major refactoring effort.3.4. Universal Configuration Backbone: HydraTo unify all three training stages and ensure a consistent, reproducible, and flexible configuration system, Hydra is recommended as the universal configuration backbone. Hydra's strength lies in its ability to compose complex configurations from smaller, reusable YAML files and to override any parameter from the command line.The proposed config/ directory structure enables this compositional approach. A complete experiment can be defined by composing defaults:config/experiment/train_foundation_h100.yamlYAMLdefaults:
  - _self_
  - data: openx_webdataset
  - model: foundation_policy_large
  - training: fsdp_bfloat16
  - hydra/launcher: slurm_h100

# Override specific parameters for this experiment
training:
  num_nodes: 8
  learning_rate: 1.0e-5
This single command, python scripts/train.py -m experiment=train_foundation_h100, would launch an 8-node training job on the H100 partition, using the large Foundation Policy model, the Open-X WebDataset, and the FSDP training configuration with bfloat16 precision. This system provides exceptional power and clarity for managing the vast hyperparameter space of a complex research project.4. System Component Design and InterfacesA clear definition of component boundaries and interfaces is essential for building a modular and maintainable system. This section outlines the high-level design and key methods for each of the three core policy components, as well as their interaction during inference.4.1. Component Overview DiagramThe LAPA system operates as a sequential pipeline. During inference, an observation and a language instruction are processed by the Foundation Policy to generate a latent action, which is then translated into a concrete robot command by the Low-Level Policy. The LAQ model is used only during its own training phase and to generate the target data for the Foundation Policy.Plaintext+-----------------+      +----------------------+      +--------------------+

| Video Frame | | Image Observation  & | | Latent Action & |
| Pair (t, t+1) | | Language Instruction | | Current Observation|
+-------+---------+      +----------+-----------+      +----------+---------+

| | |
        v                         v                         v
+-------+---------+      +----------+-----------+      +----------+---------+

| LAQ Model |----->| Foundation Policy |----->| Low-Level Policy |
| (Training Only) | | (VLM) | | (Controller) |
+-----------------+      +----------------------+      +--------------------+

| | |
        v                         v                         v
+-------+---------+      +----------+-----------+      +----------+---------+

| Latent Action | | Latent Action | | 7-DOF Robot |
| (for training | | (Prediction) | | Command |
| Foundation Pol.)| +----------------------+      +--------------------+
+-----------------+
4.2. Interface Definition: LAQModelThe Latent Action Quantization (LAQ) model is responsible for learning a compressed, discrete representation of actions from pairs of video frames. It is fundamentally an autoencoder with a vector-quantized bottleneck.Plain-text Interface: laq.models.LAQModelPlaintextclass LAQModel(torch.nn.Module):
    """
    Learns to encode actions from video frame pairs into discrete latent codes.
    """
    def __init__(self, encoder_config: Dict, vq_config: Dict, decoder_config: Dict):
        """
        Initializes the visual encoder, vector quantizer, and decoder.
        - encoder_config: Defines the architecture for processing frame pairs (e.g., a TimeSformer or 3D ResNet).
        - vq_config: Defines the Vector-Quantization module, including codebook size (e.g., 8) and number of quantizers (e.g., 4).
        - decoder_config: Defines the architecture to reconstruct an action representation from the quantized latent.
        """
        #... implementation...

    def encode(self, frame_t: torch.Tensor, frame_t_plus_1: torch.Tensor) -> torch.Tensor:
        """
        Encodes a pair of frames into discrete latent indices.
        Args:
            frame_t: Batch of frames at time t. Shape:
            frame_t_plus_1: Batch of frames at time t+1. Shape:
        Returns:
            A tensor of discrete latent indices. Shape: (e.g.,)
        """
        #... implementation...

    def decode(self, latent_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes discrete latent indices back into a continuous action representation.
        Args:
            latent_indices: A tensor of discrete latent indices. Shape:
        Returns:
            The reconstructed continuous action representation.
        """
        #... implementation...

    def forward(self, frame_t: torch.Tensor, frame_t_plus_1: torch.Tensor) -> Tuple:
        """
        The main training forward pass.
        Returns:
            A tuple containing (reconstruction_loss, commitment_loss) for the VQ-VAE objective.
        """
        #... implementation...
4.3. Interface Definition: FoundationPolicyModelThe Foundation Policy is the central component of the LAPA system. It is a large Vision-Language-Action model that maps from visual observations and natural language commands to the discrete latent action space learned by the LAQ model.Plain-text Interface: foundation.models.FoundationPolicyModelPlaintextclass FoundationPolicyModel(torch.nn.Module):
    """
    A Vision-Language-Action model that predicts latent actions from images and language instructions.
    """
    def __init__(self, vision_encoder_config: Dict, language_model_config: Dict, action_head_config: Dict):
        """
        Initializes the core components of the VLM.
        - vision_encoder_config: Defines the visual backbone (e.g., a pre-trained ViT).
        - language_model_config: Defines the text encoder (e.g., a pre-trained T5 or Llama).
        - action_head_config: Defines the multimodal fusion and prediction head.
        """
        #... implementation...

    def forward(self, image_observation: torch.Tensor, language_instruction: torch.Tensor) -> torch.Tensor:
        """
        Predicts logits over the discrete latent action space.
        Args:
            image_observation: Batch of current visual observations. Shape:
            language_instruction: Batch of tokenized language instructions. Shape:
        Returns:
            Logits for each of the latent action tokens. Shape: (e.g.,)
        """
        #... implementation...
4.4. Interface Definition: LowLevelPolicyModelThe Low-Level Policy acts as the final controller, translating the abstract, discrete latent action from the Foundation Policy into a continuous, executable command for the robot hardware.Plain-text Interface: low_level.models.LowLevelPolicyModelPlaintextclass LowLevelPolicyModel(torch.nn.Module):
    """
    Translates latent actions into concrete, continuous robot commands.
    """
    def __init__(self, latent_encoder_config: Dict, observation_encoder_config: Dict, policy_head_config: Dict):
        """
        Initializes the policy components.
        - latent_encoder_config: Defines how to embed the discrete latent indices.
        - observation_encoder_config: Defines how to process the current robot observation (image and/or proprioception).
        - policy_head_config: Defines the final network (e.g., MLP or Diffusion Decoder) that outputs the robot command.
        """
        #... implementation...

    def forward(self, latent_indices: torch.Tensor, current_observation: torch.Tensor) -> torch.Tensor:
        """
        Generates a continuous robot command.
        Args:
            latent_indices: The discrete action predicted by the Foundation Policy. Shape:
            current_observation: The current, high-frequency robot observation. Shape:
        Returns:
            A continuous 7-DOF robot command. Shape: (x, y, z, roll, pitch, yaw, gripper)
        """
        #... implementation...
An important design consideration emerges from these interfaces. The Foundation Policy outputs a very low-bandwidth signal: 4 integers between 0 and 7. This signal is highly compressed and abstract; it communicates what to do (the intent), but not the fine-grained details of how to do it in the current physical context. For example, the latent code for "grasp" does not contain the precise 3D coordinates of the target object relative to the gripper. Therefore, for the Low-Level Policy to generate a precise motor command, it is essential that it receives not only the abstract latent_indices but also the rich, high-frequency current_observation. This makes the Low-Level Policy more than a simple decoder; it is a latent-conditioned visuomotor controller that uses the latent code to select a behavior and the current observation to execute it precisely.4.5. Inference Sequence DiagramThe following plain-text sequence diagram illustrates the end-to-end data flow for predicting and executing a single robot action.PlaintextUser         System              Robot          FoundationPolicy   LowLevelPolicy

| | | | |
| "pick up..."| | | |
|------------>| | | |
| | get_observation() | | |
| |------------------>| | |
| | | returns image | |
| | |------------------>| |
| | forward(image, "pick up...") | |
| |-------------------------------------->| |
| | | | returns |
| | | | latent_indices |
| | | |----------------->|
| | forward(latent_indices, image) | |
| |--------------------------------------------------------->|
| | | | | returns
| | | | | robot_command
| | | | |<------------|
| | execute(robot_command) | |
| |------------------>| | |
| | | (robot arm moves) | |
| | | | |
5. Scaling the Foundation: Multi-Node Training with FSDPTraining the Foundation Policy, a large Vision-Language-Action model, on the required scale of data is the most computationally demanding aspect of the LAPA project. It will necessitate a distributed training strategy that can scale across multiple nodes and multiple GPUs per node. Given the anticipated size of the model, standard data parallelism is insufficient.5.1. The Necessity of FSDP (Fully Sharded Data Parallel)A state-of-the-art VLM can easily have tens or even hundreds of billions of parameters. The memory required to store such a model, along with its gradients and optimizer states, far exceeds the capacity of a single GPU, even the 94 GB of an NVIDIA H100.Standard DistributedDataParallel (DDP) works by replicating the entire model, gradients, and optimizer states on each GPU.14 This approach is fundamentally limited by the memory of a single device. Fully Sharded Data Parallel (FSDP) overcomes this limitation. Instead of replicating, FSDP shards the model parameters, gradients, and optimizer states across all available GPUs in the distributed process group.14During computation, each FSDP unit (typically a model layer) performs an all_gather operation to momentarily reconstruct the full parameters for its forward or backward pass, and then immediately discards the non-local parameters to free memory.16 This strategy dramatically reduces the peak memory footprint on each GPU, making it possible to train models that are orders of magnitude larger than what would fit on a single device. For the Foundation Policy, FSDP is not an optimization but a necessity.5.2. Implementing FSDP with Lightning FabricAs established, Lightning Fabric provides the ideal framework for implementing FSDP, offering control over the training loop while managing the distributed backend. The implementation follows a clear pattern.First, the Fabric object is initialized with the FSDP strategy. The model is then initialized on a "meta" device, which allocates the model structure without consuming any memory for its parameters. This is a critical step for very large models that would not fit on a single CPU's RAM.15 The optimizer is created after the model is defined. Finally, fabric.setup() is called, which moves the model to the correct GPU devices and wraps it with the FSDP logic.A key detail for efficient FSDP is the auto_wrap_policy. Instead of wrapping the entire model as a single FSDP unit, it is far more efficient to wrap individual layers or blocks (e.g., each TransformerBlock). This allows FSDP to overlap communication (gathering parameters for layer N+1) with computation (the forward pass of layer N), hiding communication latency and improving training throughput.16 PyTorch provides policies like transformer_auto_wrap_policy specifically for this purpose.17Code Structure for FSDP Training with Fabric:Plaintextimport torch
from lightning.fabric import Fabric
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from foundation.models import FoundationPolicyModel, TransformerBlock

# Define the wrapping policy for the transformer blocks
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# 1. Initialize Fabric with the FSDP strategy and the custom wrap policy
fabric = Fabric(
    accelerator="cuda",
    devices=4,
    num_nodes=8,
    strategy=FSDPStrategy(auto_wrap_policy=auto_wrap_policy)
)
fabric.launch() # Spawns processes across all nodes/devices

# 2. Initialize the model on a 'meta' device to save memory
with fabric.init_module():
    model = FoundationPolicyModel(...)

# 3. Initialize the optimizer (must be done after model initialization)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 4. Fabric moves the model to the device, shards it with FSDP, and sets up the optimizer
model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(train_dataloader)

# 5. Standard PyTorch training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        image_obs, lang_instr, target_actions = batch
        
        predicted_logits = model(image_obs, lang_instr)
        loss = calculate_loss(predicted_logits, target_actions)
        
        # fabric.backward handles the distributed backward pass
        fabric.backward(loss)
        
        optimizer.step()
5.3. Performance Optimization on H100 GPUsTo maximize training throughput on the powerful H100 nodes at LRZ, several optimization techniques should be employed:Mixed Precision: The H100 GPUs have specialized hardware for bfloat16 (BF16) computation. Using BF16 mixed precision can provide significant speedups (up to 4x in some cases) and reduce memory consumption by nearly half compared to full FP32 precision.17 This is configured directly within the FSDPStrategy in Fabric, by passing a MixedPrecision policy object.Activation Checkpointing (Gradient Checkpointing): The activations computed during the forward pass must be stored in memory for the backward pass, consuming a large amount of GPU memory. Activation checkpointing trades computation for memory by not storing intermediate activations for certain layers. During the backward pass, it recomputes them. This allows for training much larger models or using larger batch sizes.15 This can be applied to the transformer blocks of the Foundation Policy using PyTorch's built-in checkpointing utilities.Sharding Strategy: FSDP offers several sharding strategies that provide a trade-off between memory savings and communication overhead.15FULL_SHARD: Shards model parameters, gradients, and optimizer states. This offers the maximum memory savings but incurs the most communication.SHARD_GRAD_OP: Shards gradients and optimizer states but replicates parameters within each node. It uses more memory than FULL_SHARD but reduces communication, often leading to faster training.HYBRID_SHARD: A combination of the two, performing full sharding across nodes but only sharding gradients and optimizer states within a node. This is often optimal for multi-node training where inter-node bandwidth is lower than intra-node (NVLink) bandwidth.The recommended approach is to start with SHARD_GRAD_OP and, if memory constraints are hit, fall back to FULL_SHARD.5.4. Distributed Checkpointing StrategySaving and loading model checkpoints in an FSDP context is non-trivial because each rank only holds a fraction (a shard) of the full model's state. A naive call to model.state_dict() on rank 0 would yield only a small portion of the weights.The correct procedure involves gathering the full state dictionary from all ranks onto a single rank for saving. To avoid overwhelming the GPU memory of the saving rank with the full, unsharded model, the parameters should be gathered directly to CPU memory.Recommended Checkpointing Procedure:Saving: Use PyTorch's get_model_state_dict utility from the distributed checkpointing library. Configure it with full_state_dict=True and cpu_offload=True. This function will orchestrate the all_gather operation from all GPUs, move the resulting full tensors to the CPU on rank 0, and return the complete state_dict. Rank 0 can then save this state dict to the DSS filesystem.Loading: The process is reversed. On rank 0, load the full checkpoint file from DSS into CPU memory. Then, use the set_model_state_dict utility with full_state_dict=True and broadcast_from_rank0=True. This will broadcast the full state dict from rank 0 to all other ranks, and the FSDP wrapper will handle re-sharding the parameters correctly across all GPUs.This CPU-offloading strategy is the standard and safest method for handling checkpoints of very large models, as detailed in the official PyTorch FSDP documentation.166. The LRZ Deployment and Orchestration WorkflowA robust and reproducible workflow for deploying training jobs on the LRZ cluster is paramount. This involves a standardized process for building containerized environments, submitting jobs to the Slurm workload manager, and integrating these components with the Hydra configuration system.6.1. Containerization with Enroot: A Step-by-Step GuideContainers provide a self-contained, portable, and reproducible software environment, which is essential for eliminating "works on my machine" issues in research. On the LRZ compute nodes, the mandated container runtime is Enroot, not Docker. The workflow involves building a Docker image and then converting it into Enroot's native SquashFS (.sqsh) format.18The entire process should be version-controlled within the monorepo.Define a Dockerfile: A Dockerfile will be created in the containers/ directory. It will start from an official NVIDIA PyTorch container image (e.g., nvcr.io/nvidia/pytorch:24.01-py3), which comes pre-packaged with CUDA, cuDNN, and NCCL. The Dockerfile will then copy the project source code into the image and install all Python dependencies specified in pyproject.toml.Build and Push the Docker Image (Optional but Recommended): While not strictly necessary, building the Docker image and pushing it to a container registry (like Docker Hub or a private registry) is good practice for sharing and versioning.Import to Enroot on LRZ: On an LRZ login node, the Docker image is imported into the Enroot cache using the enroot import command. This command pulls the image layers and creates a raw Enroot filesystem.Bash# On an LRZ login node
enroot import docker://<your-registry>/lapa-project:latest
Create and Export the SquashFS Image: From the imported filesystem, a final, compressed, and immutable SquashFS image is created and exported to the high-performance DSS storage. This .sqsh file is the artifact that will be used by Slurm jobs.Bash# Create a runnable container filesystem from the imported image
enroot create --name lapa_v1 <your-registry>+lapa-project+latest.sqsh

# Export the container to a portable.sqsh file on DSS
enroot export --output /dss/dssfs04/pn57pi/containers/lapa_v1.sqsh lapa_v1
This workflow, documented in LRZ and Enroot guides 18, produces a single, versionable file (lapa_v1.sqsh) that encapsulates the entire software environment, ensuring perfect reproducibility for every training run.6.2. Slurm Job Orchestration with PyxisPyxis is the Slurm plugin that bridges the gap between Slurm and Enroot. It allows users to launch jobs directly inside an Enroot container by adding specific flags to their sbatch or srun commands.21A template sbatch script, stored at slurm/train.sbatch, will be the primary mechanism for launching training jobs. This script will be highly parameterized to handle the different requirements of each training stage.Template slurm/train.sbatch:Bash#!/bin/bash
#SBATCH --job-name=lapa-training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# === LRZ MCML Partition for H100s ===
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#
# === Resource Allocation ===
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=12
#SBATCH --time=72:00:00

# === Environment Setup for Multi-Node ===
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_DISABLE=0

# === Pyxis/Enroot Container Launch ===
# The srun command will execute the python script on all allocated resources
# inside the specified container.
srun --container-image=/dss/dssfs04/pn57pi/containers/lapa_v1.sqsh \
     --container-mounts=/dss/dssfs04/pn57pi:/data \
     python scripts/train.py "$@"
Key Pyxis flags used here are:--container-image: Specifies the path to the .sqsh file on the DSS filesystem.--container-mounts: Binds directories from the host system into the container. This is essential for making datasets, checkpoint directories, and code (for development) available inside the isolated container environment. Here, the user's entire project directory on DSS is mounted to /data inside the container.6.3. Integrating Hydra with SlurmWhile launchers like hydra-submitit-launcher can automate Slurm job submission, they may not easily support the custom Pyxis flags or the specific environment variable setup required for multi-node training on LRZ. A simpler and more transparent approach is recommended: use Hydra for configuration management and a lightweight shell script to construct and submit the sbatch command.The Python training script (scripts/train.py) will be a standard Hydra application that accepts command-line overrides. The Slurm batch script will pass all of its command-line arguments ("$@") directly to the Python script. This allows for a clean and powerful workflow:Bash# Submit a job from the command line
sbatch slurm/train.sbatch experiment=train_foundation_h100 training.num_nodes=8
This command submits the job defined in train.sbatch. Inside the srun command, the arguments experiment=train_foundation_h100 training.num_nodes=8 are passed to python scripts/train.py, allowing Hydra to configure the run correctly inside the container. This method is robust, easy to debug, and keeps the full power of Hydra's command-line interface.To streamline job submission further, the following table provides recommended Slurm configuration profiles for various tasks within the LAPA project.TaskPartitionNodesGPUs/NodeCPUs/GPUEst. TimeKey Slurm FlagsLAQ Training (Debug)lrz-v100x211801:00:00--partition=lrz-v100x2 --gres=gpu:1LAQ Training (Full)lrz-a100-80x4141224:00:00--partition=lrz-a100-80x4 --gres=gpu:4Foundation Policy (Multi-Node)mcml-hgx-h100-94x4842472:00:00-p mcml-hgx-h100-94x4 -q mcml --nodes=8 --gpus-per-node=4Low-Level Policy Traininglrz-a100-80x4111208:00:00--partition=lrz-a100-80x4 --gres=gpu:1Ultimately, this entire workflow—from the Dockerfile that defines the environment, to the Enroot commands that build the immutable container, to the Slurm scripts that define the hardware allocation—constitutes a set of version-controlled "reproducibility artifacts." By committing these infrastructure-as-code components to the monorepo alongside the model code, the project achieves a far higher standard of scientific reproducibility. A future researcher can check out a specific commit and perfectly recreate not just the code, but the entire computational environment and job configuration used to generate a result.7. Inference and Future DirectionsWhile the primary focus of this blueprint is on establishing a scalable training infrastructure, it is also important to consider the final deployment and inference pipeline, as well as pathways for future optimization.7.1. A Modular Inference PipelineThe three-stage architecture of LAPA lends itself naturally to a modular inference pipeline. The recommended implementation is a single Python class, LAPAPolicy, that encapsulates the entire end-to-end logic.This class would be initialized with the file paths to the three trained model checkpoints (LAQ, Foundation, and Low-Level). Upon initialization, it would load the weights for each component into memory. The core functionality would be a predict_action method that orchestrates the data flow as defined in the sequence diagram (Section 4.5).Plain-text Interface: inference.LAPAPolicyPlaintextclass LAPAPolicy:
    def __init__(self, foundation_ckpt_path: str, low_level_ckpt_path: str, device: str = "cuda"):
        # Load the Foundation and Low-Level policy models from checkpoints
        self.foundation_policy = self._load_foundation_model(foundation_ckpt_path).to(device)
        self.low_level_policy = self._load_low_level_model(low_level_ckpt_path).to(device)
        self.device = device
        # Note: The LAQ model is not needed for inference.

    def predict_action(self, image_observation: np.ndarray, language_instruction: str) -> np.ndarray:
        """
        Takes a raw image and language string and returns a 7-DOF robot command.
        """
        # 1. Preprocess inputs (convert to tensors, tokenize text)
        image_tensor, instruction_tensor = self._preprocess(image_observation, language_instruction)
        image_tensor = image_tensor.to(self.device)
        instruction_tensor = instruction_tensor.to(self.device)

        with torch.no_grad():
            # 2. Get latent action from Foundation Policy
            action_logits = self.foundation_policy(image_tensor, instruction_tensor)
            latent_indices = torch.argmax(action_logits, dim=-1)

            # 3. Get robot command from Low-Level Policy
            robot_command_tensor = self.low_level_policy(latent_indices, image_tensor)

        # 4. Postprocess output (convert to numpy array)
        return robot_command_tensor.cpu().numpy()
This design provides a clean, high-level API for integrating the trained LAPA policy into a larger robotics control stack. Each component can be updated independently by simply pointing the constructor to a new checkpoint file, facilitating continuous improvement and experimentation.7.2. Path to Real-Time PerformanceThe modular inference pipeline, while flexible, involves forward passes through two large neural networks (Foundation and Low-Level). For real-time robotic control, where action frequencies of 10 Hz or higher are often required, the combined latency of these passes may be too high. When the project moves from a research prototype to a performance-critical deployment, several optimization strategies should be considered:Model Compilation: Tools like torch.compile() 23, NVIDIA TensorRT, or ONNX Runtime can significantly accelerate inference. These compilers analyze the model's computational graph and fuse operations, optimize kernel execution, and quantize weights to lower precision (e.g., INT8), often resulting in substantial speedups with minimal loss in accuracy.Model Distillation: A smaller, single, end-to-end student model could be trained to mimic the behavior of the larger, three-stage teacher pipeline. The student model would be trained on a dataset of (image, instruction, final robot command) triplets generated by the teacher. This can consolidate the computation into a single, faster forward pass, at the cost of some performance and the modularity of the original system.Architectural Optimization: The architectures of the Foundation and Low-Level policies can be optimized for inference speed, for example by using more efficient vision backbones or smaller language models.7.3. Open Questions and Long-Term VisionThis technical blueprint provides a robust and scalable foundation for the LAPA project. The proposed architecture—a modular monorepo, a WebDataset-based I/O pipeline, a hybrid training framework, and a containerized HPC deployment workflow—is designed to be flexible enough to accommodate the evolving needs of a long-term research endeavor.The choices made, such as using Lightning Fabric for the Foundation Policy, are deliberately forward-looking, ensuring that the system can adapt to new research in large-model training without requiring fundamental architectural changes. The emphasis on version-controlled reproducibility artifacts (code, configs, containers, job scripts) establishes a rigorous foundation for collaborative and verifiable scientific progress.The system is designed to facilitate answers to the key open questions of the project. The scalability of the FSDP training pipeline allows for exploration of ever-larger models. The flexibility of the Hydra configuration system enables rapid and extensive hyperparameter sweeps. The modularity of the components allows for swapping out different model architectures (e.g., a diffusion-based Low-Level Policy) to test new hypotheses. Ultimately, this blueprint describes not a static, final system, but a dynamic and powerful research platform engineered for continuous discovery and innovation in the field of robot learning.