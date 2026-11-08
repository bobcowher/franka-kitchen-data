#!/bin/bash
# Push the demo shards to the training server. --partial: rerun after a dropped
# connection and rsync resumes rather than starting the file over.
set -euo pipefail

REMOTE=lab:/data/datasets/franka-kitchen-v2/

ssh "${REMOTE%%:*}" mkdir -p "${REMOTE#*:}"
rsync -a --partial --info=progress2 "$(dirname "$0")/../dataset/" "$REMOTE"
