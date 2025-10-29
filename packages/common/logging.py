"""
Logging utilities for LAPA project.

Implements comprehensive logging helpers as specified in PLAN.md Task 0.5.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for WandB compatibility
plt.style.use('default')
sns.set_palette("husl")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup basic Python logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def log_reconstruction_images(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    step: int,
    max_images: int = 8
) -> Dict[str, Any]:
    """
    Create reconstruction visualization for WandB logging.
    
    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        step: Training step
        max_images: Maximum number of images to log
        
    Returns:
        Dictionary with image data for WandB
    """
    import wandb
    
    # Take only the first max_images
    B = min(original.shape[0], max_images)
    original = original[:B]
    reconstructed = reconstructed[:B]
    
    # Convert from [-1, 1] to [0, 1] for visualization
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # Clamp to valid range
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # Create image pairs for WandB
    images = []
    for i in range(B):
        # Original image
        orig_img = original[i].permute(1, 2, 0).cpu().numpy()
        orig_img = (orig_img * 255).astype(np.uint8)
        
        # Reconstructed image
        recon_img = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        recon_img = (recon_img * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        comparison = np.concatenate([orig_img, recon_img], axis=1)
        
        images.append(wandb.Image(
            comparison,
            caption=f"Original vs Reconstructed {i+1}"
        ))
    
    return {
        "reconstruction_images": images,
        "step": step
    }


def compute_codebook_utilization(
    codebook_indices: torch.Tensor,
    vocab_size: int,
    num_tokens: int
) -> Dict[str, float]:
    """
    Compute codebook utilization metrics.
    
    Args:
        codebook_indices: Indices from quantizer [B, num_tokens]
        vocab_size: Size of vocabulary
        num_tokens: Number of token positions
        
    Returns:
        Dictionary with utilization metrics
    """
    # Flatten all indices
    all_indices = codebook_indices.flatten().cpu().numpy()
    
    # Count usage per token position
    utilization_per_token = []
    for token_idx in range(num_tokens):
        token_indices = codebook_indices[:, token_idx].cpu().numpy()
        unique_indices = len(np.unique(token_indices))
        utilization = unique_indices / vocab_size
        utilization_per_token.append(utilization)
    
    # Overall utilization
    unique_indices = len(np.unique(all_indices))
    overall_utilization = unique_indices / vocab_size
    
    # Convert list to individual metrics for each token position
    metrics = {
        "codebook_utilization_overall": overall_utilization,
        "unique_codes_used": unique_indices,
        "total_vocab_size": vocab_size
    }
    
    # Add per-token utilization as individual metrics
    for i, util in enumerate(utilization_per_token):
        metrics[f"codebook_utilization_token_{i}"] = util
    
    return metrics


def compute_perplexity(codebook_indices: torch.Tensor) -> float:
    """
    Compute perplexity (entropy) of codebook usage.
    
    Args:
        codebook_indices: Indices from quantizer [B, num_tokens]
        
    Returns:
        Perplexity value
    """
    # Flatten all indices
    all_indices = codebook_indices.flatten().cpu().numpy()
    
    # Compute probability distribution
    unique_indices, counts = np.unique(all_indices, return_counts=True)
    probabilities = counts / len(all_indices)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Perplexity is 2^entropy
    perplexity = 2 ** entropy
    
    return perplexity


def log_codebook_heatmap(
    codebook_indices: torch.Tensor,
    vocab_size: int,
    num_tokens: int,
    step: int
) -> Dict[str, Any]:
    """
    Create codebook usage data for WandB logging.
    
    Args:
        codebook_indices: Indices from quantizer [B, num_tokens]
        vocab_size: Size of vocabulary
        num_tokens: Number of token positions
        step: Training step
        
    Returns:
        Dictionary with usage data for WandB
    """
    # Create usage count matrix
    usage_matrix = np.zeros((num_tokens, vocab_size))
    
    for token_idx in range(num_tokens):
        token_indices = codebook_indices[:, token_idx].cpu().numpy()
        unique_indices, counts = np.unique(token_indices, return_counts=True)
        usage_matrix[token_idx, unique_indices] = counts
    
    # Create a simple table for WandB
    table_data = []
    for token_idx in range(num_tokens):
        for code_idx in range(vocab_size):
            table_data.append([
                f"Token_{token_idx}",
                f"Code_{code_idx}",
                int(usage_matrix[token_idx, code_idx])
            ])
    
    import wandb
    table = wandb.Table(
        columns=["Token", "Code", "Usage_Count"],
        data=table_data
    )
    
    return {
        "codebook_usage_table": table,
        "step": step
    }


def log_training_metrics(
    losses: Dict[str, float],
    codebook_indices: torch.Tensor,
    vocab_size: int,
    num_tokens: int,
    step: int
) -> Dict[str, Any]:
    """
    Compute and log comprehensive training metrics.
    
    Args:
        losses: Dictionary of loss values
        codebook_indices: Indices from quantizer [B, num_tokens]
        vocab_size: Size of vocabulary
        num_tokens: Number of token positions
        step: Training step
        
    Returns:
        Dictionary with all metrics for WandB logging
    """
    metrics = losses.copy()
    
    # Add codebook utilization metrics
    utilization_metrics = compute_codebook_utilization(
        codebook_indices, vocab_size, num_tokens
    )
    metrics.update(utilization_metrics)
    
    # Add perplexity
    metrics["perplexity"] = compute_perplexity(codebook_indices)
    
    # Add step
    metrics["step"] = step
    
    return metrics