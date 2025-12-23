#!/bin/bash
# scripts/enroot_wrapper.sh
# Wrapper script for Submitit: launches commands inside Enroot container
# with mirror-mounted paths to preserve pickle compatibility.
#
# Submitit serializes Python objects with pickle, embedding absolute paths.
# Mirror mounting ensures paths match between host and container.
#
# Usage (automatic via Hydra submitit):
#   ./scripts/enroot_wrapper.sh python train.py ...
#
# Configuration via environment variables:
#   HLRP_CONTAINER_IMAGE - Path to .sqsh container image
#   HLRP_DSS_ROOT - DSS storage root to mirror-mount (default: auto-detect)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Container image location (override via HLRP_CONTAINER_IMAGE)
# Default: user's workspace container
CONTAINER_IMAGE="${HLRP_CONTAINER_IMAGE:-/dss/dsshome1/00/go98qik2/workspace/containers/lam.sqsh}"

# Detect workspace root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# DSS roots to mount (mirror mount for pickle compatibility)
# These ensure /dss/... paths work identically in host and container
DSS_HOME_ROOT="/dss/dsshome1"
DSS_MCML_ROOT="/dss/dssmcmlfs01"

# ============================================================================
# Container Detection
# ============================================================================

# Check if already inside a container
if [ -f "/.enroot" ] || [ -f "/.dockerenv" ]; then
    # Already inside container - execute directly
    exec "$@"
fi

# ============================================================================
# Validation
# ============================================================================

# Verify container exists
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container image not found: $CONTAINER_IMAGE" >&2
    echo "" >&2
    echo "Set HLRP_CONTAINER_IMAGE environment variable to specify container path." >&2
    echo "Example: export HLRP_CONTAINER_IMAGE=/path/to/container.sqsh" >&2
    exit 1
fi

# ============================================================================
# Launch Container
# ============================================================================

echo "=== Enroot Wrapper ===" >&2
echo "Container: $CONTAINER_IMAGE" >&2
echo "Workspace: $WORKSPACE_ROOT" >&2
echo "Command: $*" >&2
echo "======================" >&2

# Build mount arguments
# Mirror mount DSS roots so pickled paths resolve correctly
MOUNT_ARGS=""

# Mount DSS home (contains user workspaces)
if [ -d "$DSS_HOME_ROOT" ]; then
    MOUNT_ARGS="$MOUNT_ARGS --mount ${DSS_HOME_ROOT}:${DSS_HOME_ROOT}"
fi

# Mount DSS MCML storage (contains datasets, checkpoints)
if [ -d "$DSS_MCML_ROOT" ]; then
    MOUNT_ARGS="$MOUNT_ARGS --mount ${DSS_MCML_ROOT}:${DSS_MCML_ROOT}"
fi

# Launch inside container
# --rw: Allow writes (for outputs, checkpoints)
# Mirror mounts ensure paths match between host and container
exec enroot start \
    $MOUNT_ARGS \
    --rw \
    "$CONTAINER_IMAGE" \
    "$@"
