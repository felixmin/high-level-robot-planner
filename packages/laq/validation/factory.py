"""
Factory for creating validation strategies from configuration.
"""

from typing import Any, Dict, List, Optional, Type

from .core import ValidationStrategy
from .visualization import BasicVisualizationStrategy
from .analysis import (
    LatentTransferStrategy,
    ClusteringStrategy,
    CodebookHistogramStrategy,
    LatentSequenceHistogramStrategy,
    AllSequencesHistogramStrategy,
)
from .scatter import (
    ActionTokenScatterStrategy,
    ActionSequenceScatterStrategy,
    TopSequencesScatterStrategy,
    StateSequenceScatterStrategy,
)
from .flow import FlowVisualizationStrategy


# Strategy type registry
STRATEGY_REGISTRY: Dict[str, Type[ValidationStrategy]] = {
    "basic": BasicVisualizationStrategy,
    "basic_visualization": BasicVisualizationStrategy,
    "latent_transfer": LatentTransferStrategy,
    "clustering": ClusteringStrategy,
    "codebook_histogram": CodebookHistogramStrategy,
    "sequence_histogram": LatentSequenceHistogramStrategy,
    "all_sequences_histogram": AllSequencesHistogramStrategy,
    "action_token_scatter": ActionTokenScatterStrategy,
    "action_sequence_scatter": ActionSequenceScatterStrategy,
    "top_sequences_scatter": TopSequencesScatterStrategy,
    "state_sequence_scatter": StateSequenceScatterStrategy,
    "flow_visualization": FlowVisualizationStrategy,
}


def create_validation_strategies(
    config: Dict[str, Any],
    val_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ValidationStrategy]:
    """
    Create validation strategies from config.

    Args:
        config: validation.strategies config dict
        val_buckets: Optional dict of bucket definitions for visualization

    Returns:
        List of ValidationStrategy instances
    """
    strategies = []

    if not config:
        return strategies

    for instance_name, instance_config in config.items():
        if not instance_config.get("enabled", True):
            continue

        # Get strategy type (defaults to instance_name for backwards compat)
        strategy_type = instance_config.get("type", instance_name)

        if strategy_type not in STRATEGY_REGISTRY:
            print(f"Warning: Unknown strategy type '{strategy_type}' for instance '{instance_name}', skipping")
            continue

        strategy_class = STRATEGY_REGISTRY[strategy_type]

        # Build kwargs, excluding 'type' which is for routing
        kwargs = {k: v for k, v in instance_config.items() if k != "type"}
        kwargs["name"] = instance_name  # Use instance name, not type

        # Pass val_buckets to basic visualization
        if strategy_type in ("basic", "basic_visualization") and val_buckets:
            kwargs["val_buckets"] = val_buckets

        try:
            strategies.append(strategy_class(**kwargs))
        except TypeError as e:
            print(f"Error creating strategy '{instance_name}' ({strategy_type}): {e}")

    return strategies
