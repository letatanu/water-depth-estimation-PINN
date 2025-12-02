#!/usr/bin/env bash
set -e
DEVICES="6,7" 

## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodPINN/:/working \
  -v /data/nhl224/code/semantic_2D/FloodPINN/data:/data \
  "${DOCKER_IMAGE}"
