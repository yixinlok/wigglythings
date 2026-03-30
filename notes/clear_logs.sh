#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "logs directory not found: $LOG_DIR" >&2
  exit 1
fi

count_before=$(find "$LOG_DIR" -mindepth 1 | wc -l | tr -d ' ')

# Remove all files/subdirectories inside logs, but keep logs itself.
find "$LOG_DIR" -mindepth 1 -exec rm -rf -- {} +

echo "Cleared logs folder: $LOG_DIR"
echo "Removed entries: $count_before"
