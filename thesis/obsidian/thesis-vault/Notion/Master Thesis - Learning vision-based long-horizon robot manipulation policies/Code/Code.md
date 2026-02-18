---
notion-id: 2b820c92-0436-8091-9608-f28b4b8172bf
---
last todos

- check out cosmos policy
- improved dataloader (review commits)
- 

> [!note]+ metrics toodo
> ● Excellent! I found the checkpoints and have a comprehensive view of the validation infrastructure. Let me
> also look at the model architecture and the OXE adapter to understand what metadata is available.
> 
> ● Searched for 3 patterns, read 4 files (ctrl+o to expand)
> 
> ● Now I have a comprehensive picture. Let me summarize what I found and provide a structured analysis of ideas
> for validating action token generalization.
> 
> ---
> 
> ## Summary: Checkpoints Found
> Checkpoint: 80k (with flow)
> Path:
> /mnt/data/workspace/runs/hlrp/2026-02-02_10-58-35_laq_continue_with_flow/checkpoints/laq-stepstep=080000.ckpt
> Notes: 5.9 GB, RAFT flow supervision
> ────────────────────────────────────────
> Checkpoint: 85k (standard)
> Path: /mnt/data/workspace/runs/hlrp/2026-02-02_01-10-58_laq_continue_52k/checkpoints/laq-stepstep=085000.ckpt
> Notes: 5.5 GB, no flow
> ────────────────────────────────────────
> Checkpoint: 52.5k base
> Path: /mnt/data/workspace/code/high-level-robot-planner/laq-stepstep052500.ckpt
> Notes: In repo root
> 
> Existing Validation Infrastructure
> 
> ## You already have substantial validation tooling:
> ┌───────────────────────────────┬──────────────────────────────────────┬─────────────────┐
> │ Strategy │ What it tests │ Metadata needed │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ LatentTransferStrategy │ Apply action from scene A to scene B │ frames only │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ ActionTokenScatterStrategy │ Token → 2D action mapping │ action │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ ActionSequenceScatterStrategy │ Full sequence → action mapping │ action │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ TopSequencesScatterStrategy │ Most common sequences vs actions │ action │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ StateSequenceScatterStrategy │ Token distribution in state space │ initial_state │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ SequenceExamplesStrategy │ Visual examples per token sequence │ frames │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ CodebookHistogramStrategy │ Codebook utilization │ codes │
> ├───────────────────────────────┼──────────────────────────────────────┼─────────────────┤
> │ CodebookEmbeddingStrategy │ Codebook geometry (t-SNE/UMAP) │ codes │
> └───────────────────────────────┴──────────────────────────────────────┴─────────────────┘
> 
> Ideas for Analyzing Action Token Generalization
> 
> Core Question
> 
> Do action tokens encode movements that generalize across robots/environments?
> 
> If token 5 means "move right" in language_table, it should also mean "move right" when applied to Bridge or
> RT-1 scenes.
> 
> ---
> 
> 1. Cross-Dataset Token Consistency (Quantitative)
> 
> Idea: For each token, collect ground-truth actions from multiple datasets and check if they're consistent.
> 
> Method:
> For each token t:
> actions_lt = [actions where token == t in language_table]
> actions_bridge = [actions where token == t in bridge]
> actions_rt1 = [actions where token == t in rt1]
> 
> ```plain text
> variance_within = avg(var(actions_lt), var(actions_bridge), ...)
> variance_across = var(mean(actions_lt), mean(actions_bridge), ...)
> 
> consistency_score = variance_within / variance_across
> ```
> 
> Good result: Low variance_across (same token = same movement direction across datasets)
> 
> Visualization: Per-token action histograms faceted by dataset
> 
> ---
> 
> 2. Action Direction Compass
> 
> Idea: Map each token to its mean action vector and visualize as arrows.
> 
> Method:
> 
> - For each token, compute mean (dx, dy) across all samples
> - Plot as unit vectors emanating from origin
> - Color by dataset to check consistency
> 
> Good result: Tokens form a consistent "compass" — token 3 points up, token 7 points right, regardless of
> which dataset the samples came from.
> 
> ---
> 
> 3. Cross-Dataset Latent Transfer
> 
> Extend your existing LatentTransferStrategy:
> 
> Current: Apply action from scene A to scene B within same bucket
> Proposed: Apply action from Dataset A to initial frame from Dataset B
> 
> 4. Get (frame_t, frame_t+k) from language_table → encode → token
> 5. Get frame_t' from Bridge (different scene, different robot)
> 6. Decode: frame_t' + token → predicted_frame'
> 7. Check: Does the predicted movement match the original movement direction?
> 
> Qualitative: Visual grid showing cross-dataset transfers
> Quantitative: If ground-truth actions available, measure angular error between original and transferred
> movement
> 
> ---
> 
> 8. Disentanglement Test (Token Invariance to Scene)
> 
> Core idea: A good token encodes ONLY movement, not scene or robot identity.
> 
> Test:
> 
> 9. Take many different initial frames with the SAME ground-truth action (e.g., move right 5cm)
> 10. Encode each (frame_t, frame_t+k) pair → token
> 11. Check: Do they all produce the same (or similar) token?
> 
> Metric: Token entropy given fixed action
> 
> - Low entropy = tokens are action-invariant (good)
> - High entropy = tokens encode scene information (bad)
> 
> ---
> 
> 12. Text Instruction → Token Alignment
> 
> Idea: Similar text instructions should produce similar tokens.
> 
> Method:
> 
> 13. Cluster text instructions by semantic meaning:
>     - "Push block right"
>     - "Move the red block to the right"
>     - "Slide object right"
> 14. For each cluster, collect all token sequences
> 15. Measure purity: Do similar instructions → similar tokens?
> 
> Metric: Normalized mutual information between instruction clusters and token assignments
> 
> ---
> 
> 16. Robot-Agnostic Reconstruction Test
> 
> Idea: Apply token from robot A to robot B and check if movement direction is preserved.
> 
> Setup:
> Robot A (language_table): frame_t → frame_t+k (action: move right)
> Robot B (Bridge WidowX): frame_t' (kitchen scene)
> 
> Encode: token = encode(frame_t, frame_t+k)
> Decode: pred = decode(frame_t', token)
> 
> Measure: Does pred show rightward movement?
> 
> Use optical flow on the reconstruction to measure movement direction.
> 
> ---
> 
> 17. Token Sequence Permutation Ablation
> 
> Test: What information is in the token ordering?
> 
> Original sequence: [3, 7, 2, 1]
> Permuted sequence: [1, 2, 3, 7]
> Reversed sequence: [1, 2, 7, 3]
> 
> Apply each to same initial frame and compare reconstructions.
> 
> If reconstructions are very different → sequence order matters (carries information)
> If similar → tokens are somewhat redundant
> 
> ---
> 
> 18. Action Magnitude vs Token Analysis
> 
> Question: Do tokens encode magnitude or just direction?
> 
> Test:
> 
> - Group samples by action magnitude (small, medium, large movements)
> - Check if different magnitudes → different tokens
> - Or if direction is primary and magnitude is secondary
> 
> ---
> 
> Implementation Approaches
> 
> Option A: Standalone Analysis Script
> 
> Create scripts/analyze_action_tokens.py:
> 
> - Load checkpoint
> - Stream samples from multiple OXE datasets
> - Compute cross-dataset consistency metrics
> - Generate visualizations
> - Save report
> 
> Option B: New Validation Strategies
> 
> Add to existing framework:
> 
> - CrossDatasetTokenConsistencyStrategy
> - TokenDirectionCompassStrategy
> - TextTokenAlignmentStrategy
> 
> Option C: Jupyter Notebook for Interactive Exploration
> 
> - Load model once
> - Interactively probe different hypotheses
> - Generate publication-quality figures
> 
> ---
> 
> Recommended Starting Points
> 
> 19. Quick win: Use existing ActionSequenceScatterStrategy with buckets for different datasets — see if same
> tokens appear in same regions across datasets
> 20. Cross-dataset transfer: Extend LatentTransferStrategy to work across buckets (apply language_table action
> to bridge frame)
> 21. Token compass: New visualization showing mean action direction per token, faceted by dataset
> 22. Quantitative metric: Token-action consistency score across datasets
> 
> Would you like me to implement any of these analyses? I can start with a standalone script that loads the 80k
> checkpoint and runs cross-dataset token analysis, or add new validation strategies to your existing
> framework.
> 
> ✻ Brewed for 7m 0s
> 
> ❯ What do you think about a test where you take two images, then you compute the flow between them, and
> 
> finally you take the mean of all the flow pixels?
> 
> Usually, flow is a white background because only the robot arm moves, and the robot arm has a certain
> 
> color. So, usually, the whole arm moves in a certain direction. My idea would be to just take the mean
> 
> of all the pixel color values, because then you have a single value that represents the direction in
> 
> which the arm is moving. Then we could kind of check whether tokens map to the same color or not, or to
> 
> the same movement direction. I think yeah, ActionSequenceScatter is also an option, but maybe that
> 
> color-based version is even better. I don't know, what do you think about that? Color might generalize
> 
> better.
> 
> ● This is a really good idea, and I think it's actually superior to relying on ground-truth action labels for
> several reasons:
> 
> Why Mean Flow is Better
> 
> 23. Universal across datasets
> - Ground-truth actions have different coordinate systems (world frame, end-effector frame, joint space)
> - Bridge uses 3D world_vector, language_table uses 2D, RT-1 uses different scaling
> - Flow is purely visual — "pixels moved this way" — same semantics everywhere
> 24. No metadata dependency
> - Many datasets don't have clean action labels
> - YouTube videos have no actions at all
> - Flow works on any video
> 25. Matches what the model sees
> - The LAQ encoder processes pixels, not action vectors
> - Flow captures the visual signal the model is actually learning from
> - If tokens should encode movement, they should correlate with visual movement
> 26. Robot-agnostic
> - A WidowX arm moving right and a Franka arm moving right produce similar flow patterns
> - The coordinate systems are irrelevant — it's all "rightward pixel motion"
> 
> You Already Have the Infrastructure
> 
> Your model has a RAFT flow teacher:
> self.flow_teacher = RAFTTeacher(...)  # Computes GT flow
> self.flow_decoder = FlowDecoder(...)   # Predicts flow from tokens
> 
> So you can:
> 
> 27. Use flow_teacher.compute_flow(frame_t, frame_t+k) → get flow field [B, 2, H, W]
> 28. Take flow.mean(dim=[-2, -1]) → mean flow vector [B, 2]
> 29. This gives (mean_dx, mean_dy) — the average movement direction
> 
> Refinements to Consider
> 
> Simple mean might wash out signal if background dominates:
> 
> # Option 1: Simple mean
> 
> mean_flow = flow.mean(dim=[-2, -1])  # [B, 2]
> 
> # Option 2: Magnitude-weighted mean (emphasize where movement is)
> 
> magnitude = (flow[:, 0]**2 + flow[:, 1]**2).sqrt()  # [B, H, W]
> weights = magnitude / magnitude.sum(dim=[-2, -1], keepdim=True)
> weighted_flow = (flow * weights.unsqueeze(1)).sum(dim=[-2, -1])  # [B, 2]
> 
> # Option 3: Threshold (only count significant motion)
> 
> mask = magnitude > threshold
> mean_flow = (flow * mask.unsqueeze(1)).sum(dim=[-2,-1]) / mask.sum(dim=[-2,-1])
> 
> For robot videos, the magnitude-weighted version is probably best — it naturally focuses on the moving arm
> and ignores static background.
> 
> Proposed Analysis: Flow-Token Scatter
> 
> For each sample:
> 1. Compute flow between frame_t and frame_t+k
> 2. Get mean flow direction (dx, dy) — magnitude-weighted
> 3. Get token from model
> 
> Plot: Scatter of (dx, dy) colored by token
> 
> If tokens generalize: Same token → same flow direction region, regardless of dataset
> 
> Bonus: Convert (dx, dy) to angle for a 1D histogram per token — should see tight angular distributions
> 
> Comparison with ActionSequenceScatter
> ┌───────────────────┬──────────────────┬─────────────────────────┐
> │      Aspect       │ Action Metadata  │        Mean Flow        │
> ├───────────────────┼──────────────────┼─────────────────────────┤
> │ Availability      │ Some datasets    │ All datasets            │
> ├───────────────────┼──────────────────┼─────────────────────────┤
> │ Coordinate system │ Varies           │ Pixel space (universal) │
> ├───────────────────┼──────────────────┼─────────────────────────┤
> │ What it measures  │ Commanded action │ Actual visual motion    │
> ├───────────────────┼──────────────────┼─────────────────────────┤
> │ Robot-agnostic    │ No               │ Yes                     │
> ├───────────────────┼──────────────────┼─────────────────────────┤
> │ Noise             │ Clean labels     │ Some background noise   │
> └───────────────────┴──────────────────┴─────────────────────────┘
> Mean flow is more universal and directly measures what we care about — did the visual scene change in a
> consistent way for this token?
> 
> Quick Implementation Sketch
> 
> class FlowTokenScatterStrategy(MetadataScatterStrategy):
> """Scatter plot of mean optical flow colored by token."""
> 
> ```plain text
>   def required_metadata(self) -> List[str]:
>       return []  # No metadata needed — uses flow teacher
> 
>   def run(self, cache, pl_module, trainer, metric_suffix=""):
>       # Get frame pairs from cache
>       frames = cache.get_all_frames()  # [N, 2, C, H, W]
>       codes = cache.get_codes()
> 
>       # Compute flow for all pairs
>       with torch.no_grad():
>           frame_t = frames[:, 0]
>           frame_tk = frames[:, 1]
>           flow = pl_module.model.flow_teacher.compute_flow(frame_t, frame_tk)
> 
>       # Magnitude-weighted mean
>       magnitude = (flow[:, 0]**2 + flow[:, 1]**2).sqrt()
>       weights = magnitude / (magnitude.sum(dim=[-2,-1], keepdim=True) + 1e-8)
>       mean_flow = (flow * weights.unsqueeze(1)).sum(dim=[-2,-1])  # [N, 2]
> 
>       # Scatter: mean_flow colored by first token
>       tokens = [c[0].item() for c in codes]
>       # ... create scatter plot ...
> ```
> 
> Would you like me to implement this as a new validation strategy? It would be a clean addition to [scatter.py](http://scatter.py/)
> and would work on all datasets without requiring action metadata.

[[2026-01-27]]

[[2026-01-22]]

# Todos



codex added a digusting af regex matching to find the right dataset in local oxe. fix that by using the same dataset remote and locally


check impact of shuffle buffer size

- rewrite the experiments/benchmark_oxe file… this was form gemini… and test / benchmark the new oxe setup
    - benchmark oxe cannot run locally due to ram issues. find a solution to fix this…

> [!note]+ ask if the following is a problem or should be fixed i think for missing episode ids
> def **episode_to_pairs_generator**():
> 
> """Generator that yields frame pairs from episodes."""
> 
> *for* episode *in* ds:
> 
> *if* "episode_id" in episode:
> 
> episode_id = episode["episode_id"]
> 
> *if* isinstance(episode_id, tf.Tensor):
> 
> episode_id = episode_id.numpy()
> 
> episode_id = (
> 
> episode_id.decode("utf-8")
> 
> *if* isinstance(episode_id, bytes)
> 
> *else* str(episode_id)
> 
> )
> 
> *else*:
> 
> *# Some datasets like Bridge might not have episode_id*
> 
> *# Use a hash of the first step's observation or just a counter if needed*
> 
> *# For now, just use "unknown"*
> 
> episode_id = "unknown"
> 
> steps_list = list(episode["steps"])
> 
> n_steps = len(steps_list)
> 
> min_steps = offset + 1
> 
> *if* n_steps < min_steps:
> 
> *# Skip short episodes*
> 
> *continue*
> 
> max_start_id

> [!note]+ run the following after dataloader refactor
> Upon reviewing packages/common/adapters/oxe.py, specifically the **iter** method, there is a clear
> bottleneck that explains why num_workers=1 provides a ~1.5x speedup.
> 
> The "Hidden" CPU Work in **iter**
> While tf.data (TensorFlow) handles the loading and decoding of images efficiently in C++, your PyTorch adapter performs heavy data
> transformation in pure Python/PyTorch inside the iterator:
> 
> ```plain text
> 1 # packages/common/adapters/oxe.py
> 2
> 3 def __iter__(self) -> Iterator:
> 4     # ...
> 5     for item in tf_iter:
> 6         if self.return_metadata:
> 7             # ...
> 8             pair_np = pair_tf.numpy()
> 9
> 
> ```
> 
> 10             # CRITICAL PERFORMANCE HIT HERE:
> 11             pair_pt = (
> 12                 torch.from_numpy(pair_np).permute(3, 0, 1, 2).float() / 255.0
> 13             )
> 14             # ...
> 
> Why num_workers=1 is Faster
> 
> 1. Blocking Operations (Worker 0):
>     - Cast to Float: uint8 -> float32 expands memory usage by 4x.
>     - Normalization: Dividing every pixel by 255.0.
>     - Permutation: Reordering memory from (H, W, C) to (C, H, W).
>     - Mixing Logic: MultiOXEFramePairDataset also runs a Python loop with random number generation to mix datasets.
> When num_workers=0, the main process must stop training, perform these CPU-intensive math operations, and only then pass the data
> to the GPU. The GPU sits idle during this time.
> 2. Pipelining (Worker 1):
>     - With num_workers=1, a separate background process performs the float casting, normalization, and dataset mixing.
>     - It places the fully processed, ready-to-go Float32 tensors into a queue.
>     - The Main Process simply grabs the data from the queue and immediately starts the GPU forward pass.
>     - Result: The CPU works on Batch N+1 while the GPU trains on Batch N.
> 
> Conclusion
> Your observation is correct because the data transformation cost (uint8->float, /255) is high enough that overlapping it with the GPU
> step yields a significant gain.
> 
> Recommendation:
> Update your config to usage num_workers: 1. It is the "sweet spot" where you get parallelism without the complexity of managing
> sharding across multiple workers (which num_workers > 1 would require).
> 
> is this fixed now?? 

> [!note]+ run again and check if num workers is still 1.8x faster or not anymore… should not because gil is not blocked anymore…???
> python scripts/2_train_laq.py experiment=laq_oxe_all_val_3 cluster=mcml_h100 cluster.compute.time_limit=00:30:00 data.num_workers=0 data.samples_per_episode=0 +cluster.compute.mem_gb=300 data.shuffle_buffer=30
> 
> python scripts/2_train_laq.py experiment=laq_oxe_all_val_3 cluster=mcml_h100 cluster.compute.time_limit=00:30:00 data.num_workers=1 data.samples_per_episode=0 +cluster.compute.mem_gb=300 data.shuffle_buffer=30
> 

add size frame / size episodes in config

- improve logging
> [!note]+ data sample count
> File: packages/common/adapters/oxe.py
> 
> 1. Line 880-883 - Added collection of episode counts:
> sizes = [len(ds) for ds in self._datasets]
> episode_counts = [ds._num_episodes for ds in self._datasets]
> self._weights = self._compute_weights(self.dataset_configs, sizes, episode_counts)
> 2. Line 896 - Added episode_counts parameter to _compute_weights:
> def _compute_weights(self, datasets_config: list, sizes: list[int], episode_counts: Optional[list[int]] = None)
> 3. Lines 976-991 - Updated logging to show both:
> f" {name}: {final_weights[i]:.3f} ({mode}, episodes={episode_counts[i]:,}, pairs~{sizes[i]:,})"
> 
> The Issue
> 
> The _num_episodes value comes from here (line 321-323):
> self._num_episodes = self._builder.info.splits[
> self.split.split("[")[0]  # "train[:90%]" -> "train"
> ].num_examples
> 
> If your split is "train[:90%]", this returns the total episodes in "train", not 90% of it. So the logged episode count is incorrect for subset splits.
> 
> My implementation is correct for accessing the value, but the underlying value itself is wrong for percentage-based splits.
> 
> Want me to:
> 
> - A) Revert my changes
> - B) Fix the split parsing to show accurate subset counts
> - C) Keep as-is but add (full split) to the log to clarify it's not the subset count

> [!note]+ ideas
> 1. Single dataset first to verify throughput improves:
> python scripts/submit_job.py experiment=laq_oxe_language_table \
> data.num_workers=4
> 2. With 1 dataset, 4 workers = 4 TF pipelines (not 16).
> 3. Increase tf.data parallelism instead of PyTorch workers - the OXE adapter uses num_parallel_calls=None (AUTOTUNE). You could try explicit parallelism by adding to the config or modifying the adapter.
> 4. Use samples_per_episode=1 - reduces memory per episode significantly since it doesn't materialize full episodes.

- speed up → profiler showed that data pipeline is the bottleneck
    - fix the single pair per episode approach
        - see currently left side of screen… align claude and codex
        - are we still loading all frames from an episode form the gcs bucket after the changes? or will we only load the required ones?
==→ see current changes in the code==
    - improve multi frame pair speed
        - see currently right side of the screen
        - interleave… 
            - are we currently building this manually? yes
            - can we achieve in another way? idk yet
            - would interleave bring immediate parallelism?
            - no because of GIL 
            - what is the python global interpreter lock?… this beaks interleave at the moment
            - num_workers would spawn more python workers with independent GILs??
            - episode to pairs generator is the issue
        - interleave vs num workers
            - num workers and interleave are the solution
            - num workers: engine, interleave: transmission
        - benchmark with num_workers > 0 
        - pure tf + interleave vs num_workers > 0 + interleave
> [!note]+ explanation
> The Current Bottleneck (Python GIL)
> Right now, your data loading pipeline looks like this:
> Disk -> TF Reader (C++) -> Python Generator (GIL) -> TF Dataset -> PyTorch
> 
> - The Problem: The "Python Generator" step (resizing images, decoding strings) is written in Python. Even if TensorFlow wants
> to run faster, it has to wait for the Python Global Interpreter Lock (GIL).
> - The Worker Solution: Adding num_workers=4 spawns 4 separate Python processes. Now you have 4 GILs, so you can process 4x as
> much data.
> 
> The Pure TF Solution
> If you rewrite the generator using only TensorFlow operations (e.g., tf.image.resize, tf.strings.to_number), the pipeline
> becomes:
> Disk -> TF Reader (C++) -> TF Operations (C++) -> PyTorch
> 
> - The Benefit: The entire pipeline runs in C++. TensorFlow releases the GIL for C++ operations.
> - Why it removes the need for workers: TensorFlow's internal engine is highly multi-threaded. It can use all your CPU cores
> to load, resize, and mix data within a single Python process. You wouldn't need to pay the memory cost of spawning 4+
> separate PyTorch workers because TF is already parallelizing everything efficiently in the background.
> 
> Summary
> 
> - Interleave + Workers: Good. Parallelizes by brute force (multiple processes). High memory usage.
> - Interleave + Pure TF: Best. Parallelizes efficiently (C++ threads). Low memory usage. No GIL bottleneck.
> 
> Since you asked not to change anything yet, I won't touch the code. But replacing the Python generator with pure TF ops is the
> ultimate optimization here.
    - are we already utilizing the full network bandwith?

- vla improvement
    - do we need lr scheduling or something to make vla training work fine?
    - compare token output vs extra head → what did other people do?
    - openvla
        - similar to our approach but
        - replaces tokens instead of adding them
        - llama has no adding token functionality compared to qwen… qwen allows that… still it might be more practical to replace to keep the training efficient… investigate this!!
→ look at what other people achieved with qwen token replacement
→ see cosmos reason 2 for how they train
        - check how people finetune qwen… do some small experiments
    - hlrp vs lapa
> [!note]+ comp
> ┌────────────────────────┬─────────────────────────────────────┬─────────────────────────────────────────┐
> │ Feature                │ LAPA (Original)                     │ HLRP (High-Level Robot Planner)         │
> ├────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────┤
> │ Architectural Approach │ Extra Head (Multi-Head)             │ Token-Based (LLM-Native)                │
> │ Output Mechanism       │ Separate action_head (Linear layer) │ Extended Vocabulary (<ACTION>, <ACT_i>) │
> │ Inputs                 │ Distinct Embeddings (vte, ate, wte) │ Standard Text Tokens                    │
> │ Loss                   │ Multi-task weighted loss            │ Standard Next-Token Prediction (CLM)    │
> │ Framework              │ JAX / Flax                          │ PyTorch Lightning                       │
> └────────────────────────┴─────────────────────────────────────┴─────────────────────────────────────────┘

> [!note]+ current impl
> 1. The Core Concept: "The Body" vs. "The Head"
> 
> Imagine an LLM (like Llama 2 or Qwen) as a massive factory.
> 
> 2. The Body (The Transformer): This is 99% of the model. It takes an input (text/image) and processes it through many layers. Its job is "understanding."
>     - Output of the Body: It produces a Hidden State. This is a vector (a list of numbers) representing the model's "thought" at that specific moment.
>     - Size: Usually 4096 numbers (for a 7B model).
>     - Example: [-0.5, 2.1, 0.03, ...] (4096 numbers long).
> 3. The Head (The Translator): This is a simple linear layer at the very end. Its job is to translate that abstract "thought" vector into an actual word from the dictionary.
>     - Mechanism: It is a matrix multiplication. It compares the "thought" vector against every known word in the vocabulary.
>     - Output of the Head: A vector of Logits.
>     - Size: Vocabulary Size (e.g., 32,000 for Llama 2).
> 4. How Prediction Works (Step-by-Step)
> 
> You asked: "Is the output really a single numeric value in a vector where each entry stands for a token??"
> 
> Yes, exactly. Here is what happens in the final millisecond of the model:
> 
> 5. The Body outputs the Hidden State (vector of size 4096).
> 6. The Head multiplies this by the Vocabulary Matrix.
> 7. The Result is a massive list of scores (Logits), one for every possible token.
> 
> Visualizing the Output Vector:
> Imagine our vocabulary is just 4 words: ["Apple", "Banana", "Cat", "Dog"].
> 
> The Head outputs: [ 2.5,  0.1,  8.9,  -1.2 ]
> Indices:              0     1     2      3
> 
> Tokens:             Apple Banana Cat    Dog
> 
> How we choose:
> 
> 8. Softmax: We convert scores to probabilities (0% to 100%).
>     - [2.5, 0.1, 8.9, -1.2] becomes [0.01%, 0.0%, 99.9%, 0.0%]
> 9. Argmax (Greedy): We pick the index with the highest number.
>     - Index 2 is highest.
>     - Index 2 corresponds to "Cat".
>     - Prediction: "Cat".
> 
> ---
> 
> 10. Comparing the Architectures: HLRP vs. LAPA
> 
> This is where the two projects diverge.
> 
> Approach A: HLRP (Token-Based / "Polyglot")
> 
> HLRP treats actions exactly like words. It teaches the model a new language.
> 
> - Architecture: Standard LLM. One Head.
> - Vocabulary: The standard vocabulary is expanded.
>     - Old Vocab: 0 to 31,999 (English words).
>     - New Additions: 32,000 to 32,007 (Action Codes).
> - Prediction:
> The model outputs a vector of size 32,008.
> If the highest score is at index 32,003, the model predicted <ACT_3>.
> - Input (Indices vs Vectors): HLRP uses Indices.
>     1. The vision system (VQ-VAE) looks at the robot video.
>     2. It says: "This movement looks like movement type #5."
>     3. It gives the LLM the integer 5.
>     4. The LLM looks up the embedding vector for <ACT_5> and processes it.
> 
> Approach B: LAPA (Extra Head / "Hydra")
> 
> LAPA modifies the architecture to have multiple specialized output mechanisms.
> 
> - Architecture: Modified LLM. Two (or Three) Heads.
>     - Shared: The Transformer Body (layers 1-32) is shared.
>     - Split: At the very end, the path forks.
> - The Switch: LAPA uses an action_mask (a list of 0s and 1s) to tell the model which head to use.
>     - If mask == 0 (Text): Pass hidden state to LM Head (Size 32,000).
>     - If mask == 1 (Action): Pass hidden state to Action Head (Size 8).
> - Prediction:
> The model calculates two different probability distributions. It ignores the one that doesn't match the mask.
> 
> ---
> 
> 11. Extending HLRP to use an "Extra Head"
> 
> You asked: "How would a similar extension in hlrp look like? is it possible?"
> 
> Yes, it is possible. HLRP uses PyTorch Lightning, which is very flexible.
> 
> How to implement it:
> 
> 12. Modify the Constructor (`__init__`):
> Currently, the model has self.lm_head. You would add:
> 
> 1     # New separate head for actions
> 2     # 4096 = hidden size, 8 = number of action codes
> 3     self.action_head = nn.Linear(4096, 8)
> 
> 13. Modify the Forward Pass (`forward`):
> You need to tell the model when to use which head.
> 
> ```plain text
> 1     def forward(self, input_ids, action_masks, ...):
> 2         # 1. Run the body (Shared)
> 3         hidden_states = self.transformer(input_ids)
> 4
> 5         # 2. Compute Text Logits
> 6         text_logits = self.lm_head(hidden_states)
> 7
> 8         # 3. Compute Action Logits
> 9         action_logits = self.action_head(hidden_states)
> 
> ```
> 
> 10
> 11         # 4. Combine them (The tricky part)
> 12         # If action_mask is 1, pick from action_logits.
> 13         # If action_mask is 0, pick from text_logits.
> 14         final_output = torch.where(action_masks, action_logits, text_logits)
> 15         return final_output
> 
> What changes/breaks?
> 
> 14. Tools Break: Standard tools like model.generate() (which powers chat) expect a single vocabulary. If you have two heads with different sizes, standard generation scripts will
> crash. You would have to write a custom generation loop (manual for loop).
> 15. Loss Calculation: You need to be very careful to calculate "Action Loss" only on action tokens and "Text Loss" only on text tokens.
> 16. Embeddings: Ideally, you would also split the input embeddings (have a separate action_embedding layer) so the model doesn't get confused between the word "Apple" and "Action #5"
> if they share the same ID.
> 
> Summary Comparison
> 
> ┌────────────────────┬─────────────────────────────────────────┬──────────────────────────────────────────────────┐
> │ Feature            │ HLRP (Token-Based)                      │ LAPA (Extra Head)                                │
> ├────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
> │ Output Vector Size │ 32,008 (Massive)                        │ 32,000 (Text) AND 8 (Action)                     │
> │ Decision           │ "Which token (word OR action) is next?" │ "Which word is next?" OR "Which action is next?" │
> │ Simplicity         │ High (Works with standard tools)        │ Low (Requires custom loops)                      │
> │ Prediction         │ Predicts an Index (e.g., 32005)         │ Predicts an Index (e.g., 5)                      │
> └────────────────────┴─────────────────────────────────────────┴──────────────────────────────────────────────────┘
> 
> HLRP likely chose the Token-Based approach because it is much easier to maintain and allows them to use powerful libraries (like Flash Attention and Hugging Face Accelerate) without
> rewriting deep internal code.
    - autoregressive vs parallel decoding
        - what are we using at the moment?
        - we need parallel decoding to decode all actions at once and not autoregressively

- add lerobot eval
- adjust epoch length to one pass through the proportionate datasets if there is any
- add more oxe datasets to dataloader
- add youtube data back in



> [!note]+ Fix dataloading… only one pair per episode for now (gemini)
> This strongly suggests that your validation set is not actually "unseen" in a semantic sense.
> 
> If the latent code is constant (contains no information) and the scene is ambiguous (visual cues $x_t$ are insufficient), yet the model correctly predicts the
> specific future, there is only one possibility: The model has seen this specific scenario (or a near-duplicate) before.
> 
> The Likely Culprit: train[:90%] vs train[90%:]
> 
> You are using a standard TFDS percentage split:
> 
> 1 # config/data/laq_oxe_multi.yaml
> 2 train_split: "train[:90%]"
> 3 val_split: "train[90%:]"
> 
> In many OXE datasets (like Bridge, Language Table), data is collected in sessions or scenes.
> 
> 1. Sequential Recording: An operator performs 50 tasks in a row with the "Red Block" and "Blue Block" in roughly the same position.
> 2. Lack of Shuffling: TFDS splits are often based on the order of file storage. If the data is stored sequentially, train[90%:] might contain the last 5
> episodes of a 50-episode session.
> 3. The "Leak": The model trained on the first 45 episodes of that session. It learned: "When the red block is at pixel (100, 100) and the blue block is at (200,
> 200), the operator always grabs the red block."
> 4. Result: When it sees episode 46 (Validation), it recognizes the scene configuration (the precise arrangement of objects and background) and "recalls" the
> answer, rather than solving the ambiguity.
> 
> Why this happens even with "Ambiguity"
> To a human, "Pick Red" and "Pick Blue" look equally valid. But to a deep network, the specific noise pattern, lighting condition, or precise object coordinates
> of this specific recording session act as a unique fingerprint. The model isn't solving "Pick Red"; it's solving "Replay Session #124".
> 
> How to verify this?
> 
> 5. Visual Check: Look at a training batch and a validation batch side-by-side. You will likely see the exact same kitchen/table setup with the same objects in
> almost the same spots.
> 6. Held-Out Environment: The only way to prove the latent is necessary is to validate on a Held-Out Environment (or held-out objects) that the model has never
> seen.
>     - For Bridge, this means holding out an entire target kitchen (e.g., toykitchen2).
>     - For Language Table, this is harder as the environment is static, but you might hold out specific tasks.
> 
> Conclusion
> Your model is overfitting to the scene configurations. This confirms that:
> 
> 7. Dense Sampling (your original issue) helped the model memorize these configurations efficiently.
> 8. Random Split (even 90/10) is insufficient for validating ambiguity resolution in these datasets.
> 
> Immediate Fix:
> Apply the Sparse Sampling (max_pairs_per_episode=1) fix I provided. This makes it much harder for the model to memorize the trajectory of every single episode,
> forcing it to generalize more (though it won't strictly fix the split issue).





- use omegaconf ??



- Remove legacy dataloader
- On validation epoch end visualizations (actually also on train epoch end but here we are shuffling anyway i think) should be randomly taken from the dataset / dataloader and not the first then → Otherwise we have 10x the same image
> [!note]+ Chat proposal
> ## 3. Visualizations: don’t always use “the first batch”
> 
> > “On validation epoch end visualizations should be randomly taken… otherwise we have 10x the same image”
> 
> Agree 100%.
> 
> Right now you probably do something like:
> 
> - Take the **first batch** from `val_dataloader`,
> - Visualize some examples,
> - Each epoch, that first batch is the same indices (especially with `shuffle=False` and fixed val set) → same images every time.
> 
> Better:
> 
> - At `validation_epoch_end` (or `on_validation_epoch_end` hook):
>     - Sample a **random subset of indices** from `val_dataset` for visualization.
>     - Or: grab one random batch from a fresh iterator over `val_dataloader`.
> - You can still make it reproducible if needed by:
>     - Using a small RNG with `(global_seed, epoch)` to choose indices.
> 
> Key idea: **visualization sampling** should be **explicit and random**, not “always the first K samples”.

!![[image 6.png]]
- the overfitting currently shuffles before giving it to the dataloader to not have the first 100 pairs from a scene which might be all the same… but the question is should we shuffle at that stage
    - also intervened with this… currently for the overfitting config we dont have any validation… we limit the dataset to 10 and then split it to 10 train 0 val for example but actually for overfitting on 1, 10, 100 or training full dataset should all have the same val… actually thats probably not practical because then val takes forever even when only overfitting to 1 sample but you get the point… we want to be able to compare the val loss curves… so i think we should FIRST split train, val, test and THEN limit to 1 sample → like this we can have the same validation dataset for the different overfitting configs. also we can have a parameter limiting the val data size so that we are not using the full thing that we use later when we are in actual prod setting 
> [!note]+ chat proposal
> ## 4. Overfitting, shuffling, and consistent validation
> 
> You’ve spotted the core problem:
> 
> > “currently for the overfitting config we dont have any validation… we limit the dataset to 10 and then split it… but actually for overfitting on 1, 10, 100 or full we want the same val…”
> 
> Yes, the current order is:
> 
> 1. Build full_dataset
> 2. Apply `max_samples` (train+val together)
> 3. Split into train/val by `val_split`
> 
> So different `max_samples` → different val sets (or no val at all if small).
> 
> ### What we want instead
> 
> Conceptually:
> 
> 4. **Split first**, then **subset**:
>     - Define a **global** train/val/test split on the full dataset indices, once, using a fixed seed.
>     - Then:
>         - `train_max_samples` (or `max_samples`) only shrinks the **train** indices,
>         - `val` and `test` stay **fixed** across runs.
> 5. Optionally:
>     - Add a `val_max_samples` param to avoid long validation runs in tiny overfitting experiments.
> 
> This gives:
> 
> - Comparable val curves across:
>     - Overfit on 1,
>     - Overfit on 10,
>     - Full train.
> - Still lets you keep val small for debugging.
> 
> ### What about shuffling “before giving it to the dataloader”?
> 
> Right now in `LAQDataModule.setup` you:
> 
> - If `max_samples` is set and `sampling_strategy == "random"`:
>     - Use `random.sample(range(len(full_dataset)), num_samples)` to pick indices.
> - Then use simple `range` to split those into train/val.
> 
> In the **new logic**, I’d do:
> 
> 6. Build `base_dataset`.
> 7. Build a list of **all** indices: `all_indices = list(range(len(base_dataset)))`.
> 8. Shuffle `all_indices` *once* with `sampling_seed` (or keep them sequential).
> 9. Compute split:
> ```plain text
> 
> ```
> - `val_size = int(len(all_indices) * val_split)
> val_indices_full = all_indices[:val_size]
> train_indices_full = all_indices[val_size:]`
> - Apply `train_max_samples` / `val_max_samples` **only** to these:
> ```plain text
> 
> ```
> - `train_indices = train_indices_full[:train_max_samples] # or random subset of them
> val_indices = val_indices_full[:val_max_samples] # optional cap`
> - Wrap:
> ```plain text
> 
> ```
> 10. `self.train_dataset = Subset(base_dataset, train_indices)
> self.val_dataset = Subset(base_dataset, val_indices)`
> 
> Now:
> 
> - Shuffling at the **index** level is explicit and reproducible.
> - Train subsets do **not** affect which samples are in validation.
> 
> You can still keep `DataLoader(shuffle=True)` for the train loader; that handles per-epoch order.

# Testing

> [!note]+ Our goal testing level
> Granularity
> 
> - Fine for code / data transformations
> - Medium for model components and training steps
> - Coarse for end-to-end
> 
> Layers
> 
> - Infra
> - Data engineering
> - Train loops and model internals
> - Model behavior
> 
> Practical Guide
> 
> - Data pipeline
>     - Shapes and types as expected
>     - No Nans / Infs (esp. after norm / aug)
> - Model
>     - Forward has expected shapes
>     - No NaNs / Infs
>     - Parameters exist and require grad
> - Pipeline
>     - Single train step → Loss decrease
> 
> Tips for solo dev
> 
> - Test on cpu
> - Dont overspecify
> - Run before every commit
> - Only add tests when bugs appear
> - Dont consider coverage
> - Mock alot
> - Leave exptensive integration / system tests away
> - 