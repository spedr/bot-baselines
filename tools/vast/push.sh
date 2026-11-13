#!/usr/bin/env bash
# Copy code and data from this machine to a rented vast.ai instance, and results back.
#
#   tools/vast/push.sh code HOST PORT     # repo without data/runs/.venv (seconds)
#   tools/vast/push.sh data HOST PORT     # data per tools/vast/data_files.txt (~43 GB; resumable)
#   tools/vast/push.sh pull HOST PORT     # runs/ back here, never overwriting local files
#   tools/vast/push.sh size               # how much `data` would send
#
# HOST/PORT come from the instance's "SSH" button (ssh -p PORT root@HOST). Re-running a command
# only sends what changed, so an interrupted data copy just continues.
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE_DIR=/workspace/bot-baselines
mode=${1:?usage: push.sh code|data|pull HOST PORT | size}
FILTER=tools/vast/data_files.txt

if [[ $mode == size ]]; then
  rsync -a --dry-run --stats --filter="merge $FILTER" data/ /tmp/bb-size-probe/ | grep -E "Total file size|Number of files:"
  exit 0
fi
host=${2:?HOST}; port=${3:?PORT}
SSH="ssh -p $port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
case $mode in
  code)
    $SSH root@"$host" "mkdir -p $REMOTE_DIR"
    rsync -az --info=stats1 -e "$SSH" --delete \
      --exclude .venv/ --exclude data/ --exclude runs/ --exclude __pycache__/ --exclude .pytest_cache/ \
      --exclude '*.pyc' ./ root@"$host":$REMOTE_DIR/
    ;;
  data)
    # no -z: the fp16 tables do not compress; --partial keeps half-sent files on interruption
    rsync -a --partial --info=progress2 -e "$SSH" --filter="merge $FILTER" \
      data/ root@"$host":$REMOTE_DIR/data/
    ;;
  pull)
    rsync -a --ignore-existing --info=stats1 -e "$SSH" root@"$host":$REMOTE_DIR/runs/ runs/
    ;;
  *) echo "unknown mode $mode" >&2; exit 2;;
esac
