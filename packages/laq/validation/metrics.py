"""
Shared metric utilities for LAQ validation strategies.
"""

import torch


def compute_entropy(counts: torch.Tensor) -> float:
    """
    Compute entropy of a distribution given counts.

    Args:
        counts: Tensor of counts (e.g., cluster sizes, codebook usage)

    Returns:
        Entropy value in nats (natural log base)
    """
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Filter zeros to avoid log(0)
    return -(probs * probs.log()).sum().item()
