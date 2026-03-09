#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_claude_review.sh [options]

Run Claude in non-interactive print mode for a review prompt.

Options:
  --prompt-file PATH   Read the prompt from PATH. Use "-" to read from stdin.
  --output-file PATH   Write Claude's response to PATH as well as stdout.
  --model MODEL        Pass --model MODEL to Claude.
  --tools TOOLS        Pass --tools TOOLS to Claude. Default: "".
  --add-dir PATH       Add a directory Claude may inspect. Repeatable.
  --effort LEVEL       Pass --effort LEVEL to Claude.
  --help               Show this message.

If --prompt-file is omitted, the script reads the prompt from stdin.
EOF
}

prompt_file=""
output_file=""
model=""
tools=""
effort=""
declare -a add_dirs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt-file)
      prompt_file="${2:?missing value for --prompt-file}"
      shift 2
      ;;
    --output-file)
      output_file="${2:?missing value for --output-file}"
      shift 2
      ;;
    --model)
      model="${2:?missing value for --model}"
      shift 2
      ;;
    --tools)
      tools="${2-}"
      shift 2
      ;;
    --add-dir)
      add_dirs+=("${2:?missing value for --add-dir}")
      shift 2
      ;;
    --effort)
      effort="${2:?missing value for --effort}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

tmp_prompt=""
cleanup() {
  if [[ -n "$tmp_prompt" && -f "$tmp_prompt" ]]; then
    rm -f "$tmp_prompt"
  fi
}
trap cleanup EXIT

if [[ -z "$prompt_file" || "$prompt_file" == "-" ]]; then
  tmp_prompt="$(mktemp)"
  cat > "$tmp_prompt"
  prompt_file="$tmp_prompt"
fi

if [[ ! -f "$prompt_file" ]]; then
  echo "Prompt file not found: $prompt_file" >&2
  exit 2
fi

declare -a cmd
cmd=(claude -p --tools "$tools")

if [[ -n "$model" ]]; then
  cmd+=(--model "$model")
fi

if [[ -n "$effort" ]]; then
  cmd+=(--effort "$effort")
fi

for dir in "${add_dirs[@]}"; do
  cmd+=(--add-dir "$dir")
done

if [[ -n "$output_file" ]]; then
  "${cmd[@]}" < "$prompt_file" | tee "$output_file"
else
  "${cmd[@]}" < "$prompt_file"
fi
