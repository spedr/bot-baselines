#!/usr/bin/env bash
# Shared precompute chain: text embeddings for Cresci-2015, TwiBot-20 and TwiBot-22.
# Every step is idempotent (skips when its output + meta exist) and resumable per chunk, so the
# script can simply be re-run after an interruption. Each step is a separate memrun job, so the
# heavy-job lock is released between steps and other agents' jobs can interleave.
#   setsid nohup tools/precompute.sh > data/logs/precompute.log 2>&1 &
# Progress: `uv run bb status` or data/status/*.json; per-step logs in data/logs/.
set -u
cd "$(dirname "$0")/.."
BB=.venv/bin/bb
run() {
  local name=$1; shift
  echo "[$(date -Is)] start $name"
  tools/memrun -m "${MEM:-8G}" -- /usr/bin/time -v "$@" >> "data/logs/$name.log" 2>&1
  echo "[$(date -Is)] end $name rc=$?"
}
wait_done() {  # wait until a build step reports state=done
  until .venv/bin/python -c "import sys; from bb.status import step_is_done; sys.exit(0 if step_is_done('$1') else 1)"; do
    sleep 120
  done
}
STAGES="${STAGES:-cresci15 twibot20 twibot22}"
for s in $STAGES; do
  case $s in
    cresci15)
      run cresci15_emb_des        $BB embed cresci15 --encoder roberta-base --target des
      run cresci15_emb_tweets     $BB embed cresci15 --encoder roberta-base --target tweets --selection first20 ;;
    twibot20)
      run twibot20_emb_des_distil $BB embed twibot20 --encoder distilroberta-base --target des
      run twibot20_emb_des        $BB embed twibot20 --encoder roberta-base --target des
      run twibot20_emb_tweets     $BB embed twibot20 --encoder roberta-base --target tweets --selection all ;;
    twibot22)
      wait_done twibot22_tweets
      run twibot22_emb_des        $BB embed twibot22 --encoder roberta-base --target des
      run twibot22_emb_tweets     $BB embed twibot22 --encoder roberta-base --target tweets --selection first20 ;;
  esac
done
echo "[$(date -Is)] precompute chain finished"
