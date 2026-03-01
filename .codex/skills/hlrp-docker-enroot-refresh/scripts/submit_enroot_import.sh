#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI="docker://felixmin/hlrp:latest"
PARTITION="lrz-cpu"
QOS="cpu"
TIME_LIMIT="01:00:00"
MEMORY="128G"
CPUS="4"
MAX_PROCESSORS="4"
JOB_NAME="enroot-import-hlrp-oli"
OUTPUT="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_oli_$(date +%F_%H-%M-%S).sqsh"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_enroot_import.sh [options]

Submit the validated high-memory Enroot import job on ssh ai.

Options:
  --image-uri URI              Enroot image URI. Default: docker://felixmin/hlrp:latest
  --output PATH                Cluster output .sqsh path.
  --partition PARTITION        Slurm partition. Default: lrz-cpu
  --qos QOS                    Slurm qos. Default: cpu
  --time LIMIT                 Slurm time limit. Default: 01:00:00
  --mem MEMORY                 Slurm memory request. Default: 128G
  --cpus N                     Slurm cpus-per-task. Default: 4
  --max-processors N           ENROOT_MAX_PROCESSORS. Default: 4
  --job-name NAME              Slurm job name.
  --dry-run                    Print the ssh/sbatch command without executing it.
  -h, --help                   Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-uri)
      IMAGE_URI="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --max-processors)
      MAX_PROCESSORS="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${IMAGE_URI}" == docker://docker.io/* ]]; then
  echo "Use docker://<namespace>/<image>:<tag> for Docker Hub, not docker://docker.io/..." >&2
  exit 1
fi

OUTPUT_DIR=$(dirname "${OUTPUT}")
WRAP_RAW="set -euo pipefail; test ! -e \"${OUTPUT}\"; mkdir -p \"${OUTPUT_DIR}\"; export ENROOT_MAX_PROCESSORS=\"${MAX_PROCESSORS}\"; enroot import -o \"${OUTPUT}\" \"${IMAGE_URI}\"; ls -lh \"${OUTPUT}\""
printf -v REMOTE_CMD 'sbatch -p %q -q %q -t %q --mem=%q -c %q -J %q --wrap %q' \
  "${PARTITION}" "${QOS}" "${TIME_LIMIT}" "${MEMORY}" "${CPUS}" "${JOB_NAME}" "${WRAP_RAW}"

printf '+ ssh ai %q\n' "${REMOTE_CMD}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  ssh ai "${REMOTE_CMD}"
fi
