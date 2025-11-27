"""
LAPA LAQ Package

Stage 1: Latent Action Quantization (LAQ)
VQ-VAE that compresses frame-to-frame transitions into discrete latent codes.
"""

__version__ = "0.1.0"

from laq.task import LAQTask, separate_weight_decayable_params
from laq.callbacks import ReconstructionVisualizationCallback, EMACallback
from laq.models.latent_action_quantization import LatentActionQuantization
from laq.models.nsvq import NSVQ
from laq.models.attention import Attention, Transformer

__all__ = [
    "LAQTask",
    "separate_weight_decayable_params",
    "ReconstructionVisualizationCallback",
    "EMACallback",
    "LatentActionQuantization",
    "NSVQ",
    "Attention",
    "Transformer",
]

