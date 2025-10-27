#!/bin/bash
# Build and convert Docker image to Enroot .sqsh format for LRZ

set -e

IMAGE_NAME="lapa"
IMAGE_TAG="latest"
OUTPUT_PATH="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/containers"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile ..

echo "Saving Docker image to tar..."
docker save ${IMAGE_NAME}:${IMAGE_TAG} -o ${IMAGE_NAME}.tar

echo "Converting to Enroot sqsh format..."
enroot import -o ${OUTPUT_PATH}/${IMAGE_NAME}.sqsh dockerd://${IMAGE_NAME}:${IMAGE_TAG}

echo "Cleaning up..."
rm ${IMAGE_NAME}.tar

echo "Container built successfully: ${OUTPUT_PATH}/${IMAGE_NAME}.sqsh"
echo "You can now use it in Slurm scripts with:"
echo "  #SBATCH --container-image=${OUTPUT_PATH}/${IMAGE_NAME}.sqsh"

