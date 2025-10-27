#!/bin/bash
# Setup script for LAPA development environment

set -e

echo "================================================================"
echo "Setting up LAPA conda environment"
echo "================================================================"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - using environment.yml (MPS support)"
    ENV_FILE="environment.yml"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - using environment_cuda.yml (CUDA support)"
    ENV_FILE="environment_cuda.yml"
else
    echo "Using default environment.yml"
    ENV_FILE="environment.yml"
fi

# Create conda environment from yaml
echo "Creating conda environment 'lapa' from ${ENV_FILE}..."
conda env create -f ${ENV_FILE}

echo ""
echo "================================================================"
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate lapa"
echo ""
echo "Then install the LAPA packages in editable mode:"
echo "  pip install -e ."
echo ""
echo "Finally, verify the setup:"
echo "  python scripts/0_setup_environment.py"
echo "================================================================"

