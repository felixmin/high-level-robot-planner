"""
Analysis strategies for LAQ validation (latent space, clustering, histograms).
"""

from collections import Counter
from typing import Any, Dict, List, Optional
import io

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from einops import rearrange
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from PIL import Image

from .core import ValidationStrategy, ValidationCache
from .metrics import compute_entropy


class LatentTransferStrategy(ValidationStrategy):
    """
    Test if latent actions transfer between different scenes.

    For pairs (s_a, s_a') and (s_b, s_b'):
    1. Encode z_a = E(s_a, s_a')
    2. Apply z_a to s_b: s_b'_pred = D(s_b, z_a)
    3. Compare s_b'_pred with actual s_b' (should be different)
       and with s_a' (should be similar in "action" applied)

    This measures how "action-like" vs "state-specific" the latents are.
    Useful for comparing IID vs holdout buckets to see if actions generalize.
    """

    def __init__(
        self,
        name: str = "latent_transfer",
        enabled: bool = True,
        every_n_validations: int = 10,
        num_pairs: int = 256,
        min_samples: int = 4,  # Need at least 4 samples for 2 pairs
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_pairs = num_pairs

    def needs_caching(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Run latent transfer analysis."""
        metrics = {}

        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) < 4:
            return metrics

        # Sample pairs
        n = min(self.num_pairs, len(all_frames) // 2)
        indices = torch.randperm(len(all_frames))[:n * 2]

        # Split into source and target pairs
        source_frames = all_frames[indices[:n]]  # (s_a, s_a')
        target_frames = all_frames[indices[n:]]  # (s_b, s_b')

        pl_module.eval()
        with torch.no_grad():
            device = pl_module.device
            source_frames = source_frames.to(device)
            target_frames = target_frames.to(device)

            # Encode source pairs to get latent actions
            # Use task helper which handles raw pixels -> quantized latents
            source_latents, source_indices = pl_module.encode_latents(source_frames)  # z_a

            # Get target initial frames
            target_s0 = target_frames[:, :, 0:1]  # s_b (keep dim for concat)

            # Also encode target frames to get their true indices (for comparison)
            _, target_indices = pl_module.encode_latents(target_frames)

            # Decode: apply source latent to target initial frame
            # Use task helper which handles embedding and reshaping
            transferred_recons = pl_module.decode_with_latents(
                target_s0,
                source_latents,
            )  # s_b'_pred

            # Remove time dim if present [B, C, 1, H, W] -> [B, C, H, W]
            if transferred_recons.ndim == 5:
                transferred_recons = transferred_recons.squeeze(2)

            # Get ground truth
            target_s1_true = target_frames[:, :, 1]  # s_b' (true)
            source_s1_true = source_frames[:, :, 1]  # s_a' (true)

            # Compute transfer error (pred vs true target)
            transfer_mse = F.mse_loss(transferred_recons, target_s1_true)

            # Compute self-reconstruction error (for reference)
            self_recons = pl_module.model(target_frames, return_recons_only=True)
            self_mse = F.mse_loss(self_recons, target_s1_true)

        pl_module.train()

        # Use metric_suffix for bucket-specific logging
        metrics[f"val/latent_transfer_mse{metric_suffix}"] = transfer_mse.item()
        metrics[f"val/self_recon_mse{metric_suffix}"] = self_mse.item()
        metrics[f"val/transfer_ratio{metric_suffix}"] = transfer_mse.item() / (self_mse.item() + 1e-8)

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)
        
        # Visualize some transfers
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_transfers(
                source_frames[:4].cpu(),
                target_frames[:4].cpu(),
                transferred_recons[:4].cpu(),
                self_recons[:4].cpu(),  # True reconstruction for comparison
                source_indices[:4].cpu(),
                target_indices[:4].cpu(),
                wandb_logger,
                trainer.global_step,
            )

        return metrics
    
    def _visualize_transfers(
        self,
        source_frames: torch.Tensor,
        target_frames: torch.Tensor,
        transferred: torch.Tensor,
        true_recon: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        wandb_logger,
        global_step: int,
    ):
        """
        Visualize latent transfer results with true reconstruction comparison.

        Grid columns:
        1. s_a: Source first frame
        2. s_a': Source second frame (ground truth action result)
        3. s_b: Target first frame
        4. D(s_b, z_a): Transfer reconstruction (using source's latent)
        5. D(s_b, z_b): True reconstruction (using target's own latent)
        6. s_b': Target second frame (ground truth)
        """
        s_a = source_frames[:, :, 0]          # Source first frame
        s_a_prime = source_frames[:, :, 1]    # Source action result (GT)
        s_b = target_frames[:, :, 0]          # Target first frame
        s_b_recon_true = true_recon           # D(s_b, z_b) - true recon
        s_b_recon_transfer = transferred      # D(s_b, z_a) - transfer recon
        s_b_prime_true = target_frames[:, :, 1]  # Target GT

        num_samples = len(s_a)
        if num_samples == 0:
            return

        # Create figure
        # 6 columns of images
        fig, axes = plt.subplots(num_samples, 6, figsize=(20, 3.5 * num_samples), squeeze=False)
        
        # Column titles
        col_titles = ["s_a", "s_a'", "s_b", "D(s_b, z_a)\n(Transfer)", "D(s_b, z_b)\n(Self)", "s_b'"]
        
        for i in range(num_samples):
            # Row images (swapped 4 and 5 as requested)
            # 4: Transfer (z_a applied to s_b)
            # 5: Self (z_b applied to s_b)
            imgs = [
                s_a[i], 
                s_a_prime[i], 
                s_b[i], 
                s_b_recon_transfer[i], 
                s_b_recon_true[i], 
                s_b_prime_true[i]
            ]
            
            # Prepare token string
            tokens_a = str(source_indices[i].tolist())
            tokens_b = str(target_indices[i].tolist())
            
            row_text = f"Row {i}\nDet(a): {tokens_a}\nDet(b): {tokens_b}\nApp(b): {tokens_a}"
            
            for j, img in enumerate(imgs):
                ax = axes[i, j]
                
                # Convert [C, H, W] to [H, W, C] for imshow and clamp
                img_np = img.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                
                ax.imshow(img_np)
                ax.axis('off')
                
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=12)
                
                if j == 0:
                    # Add text to the left of the first image of the row
                    ax.text(-0.2, 0.5, row_text, transform=ax.transAxes, 
                            va='center', ha='right', fontsize=10, rotation=0, family='monospace')

        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        wandb_logger.log_image(
            key="val/latent_transfer",
            images=[img],
            caption=[f"Step {global_step}"],
        )
        plt.close(fig)


class ClusteringStrategy(ValidationStrategy):
    """
    Analyze latent action distribution via clustering.

    - Collect latent codes from validation set
    - Run k-means clustering
    - Log cluster statistics and example frame pairs per cluster
    """

    def __init__(
        self,
        name: str = "clustering",
        enabled: bool = True,
        every_n_validations: int = 20,
        num_samples: int = 1000,
        num_clusters: int = 16,
        num_examples_per_cluster: int = 4,
        min_samples: int = 16,  # Need at least num_clusters samples
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.num_examples_per_cluster = num_examples_per_cluster

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Run clustering analysis."""
        metrics = {}

        all_frames = cache.get_all_frames()
        all_codes = cache.get_codes()  # Use bounded codes for frame correspondence

        if all_codes is None or len(all_codes) < self.num_clusters:
            return metrics

        # Limit samples
        n = min(self.num_samples, len(all_codes))
        indices = torch.randperm(len(all_codes))[:n]
        codes = all_codes[indices]
        frames = all_frames[indices] if all_frames is not None else None

        # Flatten codes for clustering (if multi-token)
        codes_flat = codes.reshape(len(codes), -1).float()

        # K-means clustering
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(codes_flat.cpu().numpy())
            cluster_labels = torch.from_numpy(cluster_labels)

            # Compute cluster sizes
            cluster_sizes = torch.bincount(cluster_labels, minlength=self.num_clusters)

            # Log metrics with suffix
            metrics[f"val/cluster_entropy{metric_suffix}"] = compute_entropy(cluster_sizes.float())
            metrics[f"val/num_empty_clusters{metric_suffix}"] = (cluster_sizes == 0).sum().item()
            metrics[f"val/max_cluster_size{metric_suffix}"] = cluster_sizes.max().item()
            metrics[f"val/min_cluster_size{metric_suffix}"] = cluster_sizes[cluster_sizes > 0].min().item() if (cluster_sizes > 0).any() else 0

            pl_module.log_dict(metrics, sync_dist=True)
            
            # Visualize examples from each cluster
            if frames is not None:
                wandb_logger = self._get_wandb_logger(trainer)
                if wandb_logger is not None:
                    self._visualize_clusters(
                        frames, cluster_labels, wandb_logger, trainer.current_epoch
                    )
            
        except ImportError:
            print("sklearn not available for clustering analysis")

        return metrics

    def _visualize_clusters(
        self,
        frames: torch.Tensor,
        cluster_labels: torch.Tensor,
        wandb_logger,
        epoch: int,
    ):
        """Visualize example frame pairs from each cluster."""
        grids = []
        
        for cluster_id in range(self.num_clusters):
            mask = cluster_labels == cluster_id
            cluster_frames = frames[mask]
            
            if len(cluster_frames) == 0:
                continue
            
            # Get examples
            n_examples = min(self.num_examples_per_cluster, len(cluster_frames))
            examples = cluster_frames[:n_examples]
            
            # Create mini-grid: [frame_t, frame_t+offset]
            frame_t = examples[:, :, 0]
            frame_t_plus = examples[:, :, 1]
            
            imgs = torch.stack([frame_t, frame_t_plus], dim=1)
            imgs = rearrange(imgs, 'b r c h w -> (b r) c h w')
            imgs = imgs.clamp(0.0, 1.0)
            
            grid = make_grid(imgs, nrow=2, normalize=False, padding=2)
            grids.append(grid)
        
        if grids:
            # Combine all cluster grids
            combined = torch.cat(grids, dim=2)  # Horizontal concat
            
            wandb_logger.log_image(
                key="val/cluster_examples",
                images=[combined],
                caption=[f"Epoch {epoch}: Examples from {len(grids)} clusters"],
            )


class CodebookHistogramStrategy(ValidationStrategy):
    """
    Visualize codebook usage distribution as a histogram.

    Shows which codebook entries are used most/least frequently,
    helping identify if the codebook is being utilized effectively.
    """

    def __init__(
        self,
        name: str = "codebook_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True  # Need codes for histogram

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate codebook usage histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # Flatten codes to get all codebook indices used
        # all_codes shape: [N, code_seq_len] where values are codebook indices
        codes_flat = all_codes.flatten()

        # Get codebook size from model (NSVQ uses num_embeddings)
        codebook_size = pl_module.model.vq.num_embeddings

        # Count usage per codebook entry
        counts = torch.bincount(codes_flat.long(), minlength=codebook_size)

        # Compute metrics with suffix
        total_codes = counts.sum().item()
        used_codes = (counts > 0).sum().item()
        metrics[f"val/codebook_utilization{metric_suffix}"] = used_codes / codebook_size
        metrics[f"val/codebook_entropy{metric_suffix}"] = compute_entropy(counts.float())

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)

        # Create histogram visualization
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(counts, wandb_logger, trainer.global_step, codebook_size)

        return metrics

    def _create_histogram(
        self,
        counts: torch.Tensor,
        wandb_logger,
        global_step: int,
        codebook_size: int,
    ):
        """Create and log histogram of codebook usage."""
        try:
            fig, ax = plt.subplots(figsize=(10, 4))

            x = range(codebook_size)
            ax.bar(x, counts.cpu().numpy(), color='steelblue', alpha=0.8)
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Usage Count')
            ax.set_title(f'Codebook Usage Distribution (Step {global_step})')

            # Add statistics text
            used = (counts > 0).sum().item()
            ax.text(0.95, 0.95, f'Used: {used}/{codebook_size}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/codebook_histogram",
                images=[img],
                caption=[f"Step {global_step}"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: codebook_histogram visualization failed: {e}")


class LatentSequenceHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of latent token sequences (combinations).

    Since the combination space can be large (e.g., 8^4 = 4096), this strategy
    plots the top-N most frequent sequences to show if the model collapses
    to a few specific action patterns.
    """

    def __init__(
        self,
        name: str = "sequence_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_top_sequences: int = 50,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_top_sequences = num_top_sequences

    def needs_caching(self) -> bool:
        return True  # Need codes for sequence analysis

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate sequence usage histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # all_codes shape: [N, code_seq_len]
        # Convert to list of tuples for counting
        sequences = [tuple(c.tolist()) for c in all_codes]

        counter = Counter(sequences)

        # Metrics
        unique_seqs = len(counter)
        total_samples = len(all_codes)

        # Calculate entropy of sequence distribution
        counts = torch.tensor(list(counter.values()), dtype=torch.float)
        metrics[f"val/sequence_entropy{metric_suffix}"] = compute_entropy(counts)
        metrics[f"val/unique_sequences{metric_suffix}"] = unique_seqs
        metrics[f"val/total_val_samples{metric_suffix}"] = total_samples

        pl_module.log_dict(metrics, sync_dist=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(counter, wandb_logger, trainer.global_step)

        return metrics

    def _create_histogram(
        self,
        counter,
        wandb_logger,
        global_step: int,
    ):
        """Create and log histogram of top sequence usage."""
        try:
            # Get top N most common
            most_common = counter.most_common(self.num_top_sequences)
            if not most_common:
                return

            labels, values = zip(*most_common)
            # Convert tuple labels to strings "1-2-3-4"
            str_labels = ["-".join(map(str, l)) for l in labels]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(values))
            ax.bar(x, values, color='mediumpurple', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(str_labels, rotation=90, fontsize=8)
            ax.set_xlabel('Token Sequence')
            ax.set_ylabel('Count')
            ax.set_title(f'Top {len(values)} Latent Sequences (Step {global_step})')

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/sequence_histogram",
                images=[img],
                caption=[f"Step {global_step}: Distribution of top {len(values)} sequences"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: sequence_histogram visualization failed: {e}")


class AllSequencesHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of ALL latent token sequences (sorted frequency).

    This shows the "long tail" of the distribution.
    X-axis: Sequence rank (1 to N)
    Y-axis: Frequency count
    No labels on X-axis to avoid clutter.
    """

    def __init__(
        self,
        name: str = "all_sequences_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate all sequences histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return metrics

        # all_codes shape: [N, code_seq_len]
        sequences = [tuple(c.tolist()) for c in all_codes]
        counter = Counter(sequences)

        # Sort counts descending
        sorted_counts = sorted(counter.values(), reverse=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_plot(sorted_counts, wandb_logger, trainer.global_step)

        return metrics

    def _create_plot(
        self,
        counts: List[int],
        wandb_logger,
        global_step: int,
    ):
        """Create and log plot of all sequence counts."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(counts))
            ax.bar(x, counts, color='teal', width=1.0, alpha=0.8)
            # Alternatively use plot/fill_between for very dense data
            # ax.plot(x, counts, color='teal')
            # ax.fill_between(x, counts, color='teal', alpha=0.3)
            
            ax.set_xlabel('Sequence Rank')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of All {len(counts)} Unique Sequences (Step {global_step})')
            ax.set_yscale('log') # Log scale helps see the tail
            
            # Add stats
            ax.text(0.95, 0.95, f'Total Unique: {len(counts)}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key="val/all_sequences_histogram",
                images=[img],
                caption=[f"Step {global_step}: Long tail distribution (log scale)"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: all_sequences_histogram visualization failed: {e}")
