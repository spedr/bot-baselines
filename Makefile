# Convenience targets. Heavy steps go through tools/memrun (memory cap + heavy-job lock).
BB := .venv/bin/bb
MEMRUN := tools/memrun

.PHONY: install test status verify build-twibot20 build-cresci15 build-twibot22 precompute smoke

install:
	uv sync

test:
	uv run pytest

status:
	@$(BB) status

verify:
	@for d in twibot20 cresci15 twibot22; do echo "== $$d"; $(BB) verify $$d || true; done

build-twibot20:
	$(MEMRUN) -m 4G -- $(BB) build twibot20

build-cresci15:
	$(MEMRUN) -m 6G -- $(BB) build cresci15

# TwiBot-22 tweets: I/O-bound, ~1.3 GB peak, ~18 min; resumable
build-twibot22:
	$(MEMRUN) -m 4G -- $(BB) build twibot22 --step users
	$(MEMRUN) -m 2G -- $(BB) build twibot22 --step graph
	$(MEMRUN) -m 3G -- $(BB) build twibot22 --step tweets

# all text embeddings (hours; detached; idempotent and resumable)
precompute:
	setsid nohup tools/precompute.sh > data/logs/precompute.log 2>&1 < /dev/null &

smoke:
	$(BB) extscore-synth twibot20 --tag synthetic
	$(MEMRUN) -m 2G --no-lock -- .venv/bin/python -m bb.methods.logreg_ref --dataset twibot20 --seeds 0 1 2
	$(MEMRUN) -m 2G --no-lock -- .venv/bin/python -m bb.methods.logreg_ref --dataset twibot20 --seeds 0 1 2 --with-extscore synthetic
