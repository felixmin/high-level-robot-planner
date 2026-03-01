#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
WORKSPACE_ROOT=$(cd "${REPO_ROOT}/.." && pwd)

IMAGE="felixmin/hlrp:latest"
DOCKERFILE="${REPO_ROOT}/containers/Dockerfile.lerobot"
CONTEXT="${REPO_ROOT}"
PRUNE_BEFORE=1
PRUNE_AFTER=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build_push_prune.sh [options]

Build and push the HLRP image from the workstation, then leave Docker empty.

Options:
  --image IMAGE                 Docker tag to build and push.
  --dockerfile PATH             Path to Dockerfile. Defaults to containers/Dockerfile.lerobot.
  --context PATH                Docker build context. Defaults to the repo root.
  --skip-prune-before           Skip the initial Docker prune.
  --skip-prune-after            Skip the final Docker prune.
  --dry-run                     Print commands without executing them.
  -h, --help                    Show this message.
EOF
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --context)
      CONTEXT="$2"
      shift 2
      ;;
    --skip-prune-before)
      PRUNE_BEFORE=0
      shift
      ;;
    --skip-prune-after)
      PRUNE_AFTER=0
      shift
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

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

if [[ ! -f "${CONTEXT}/lerobot/pyproject.toml" ]]; then
  echo "Expected build context to contain lerobot/pyproject.toml: ${CONTEXT}" >&2
  exit 1
fi

if [[ ! -f "${CONTEXT}/lerobot/MANIFEST.in" ]]; then
  echo "Expected build context to contain lerobot/MANIFEST.in: ${CONTEXT}" >&2
  exit 1
fi

if [[ ! -d "${CONTEXT}/lerobot/src" ]]; then
  echo "Expected build context to contain lerobot/src/: ${CONTEXT}" >&2
  exit 1
fi

if [[ "${PRUNE_BEFORE}" -eq 1 ]]; then
  run docker system prune -a --volumes -f
  run docker builder prune -a -f
fi

run docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${CONTEXT}"
run docker push "${IMAGE}"

if [[ "${PRUNE_AFTER}" -eq 1 ]]; then
  run docker system prune -a --volumes -f
  run docker builder prune -a -f
fi

run docker system df
