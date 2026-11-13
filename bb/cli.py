"""Command-line entry point: ``uv run bb <command> ...`` (or ``python -m bb``).

Commands
  bb build twibot20 [--force]
  bb build cresci15 [--force]
  bb build twibot22 --step users|graph|tweets [--force]
  bb embed <dataset> --encoder roberta-base --target des|tweets [--selection first20|all] [...]
  bb status [--json]               all data/status/*.json
  bb verify <dataset>              re-check counts/invariants of built artifacts
  bb extscore-synth <dataset> [--tag T] synthetic ExtScore column for tests/smoke runs
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("dataset", choices=["twibot20", "twibot22", "cresci15"])
    b.add_argument("--step", default=None)
    b.add_argument("--force", action="store_true")

    e = sub.add_parser("embed")
    e.add_argument("dataset")
    e.add_argument("--encoder", default="roberta-base")
    e.add_argument("--target", choices=["des", "tweets"], required=True)
    e.add_argument("--selection", default="first20", help="first20 | all | first<k>")
    e.add_argument("--max-tokens", type=int, default=512)
    e.add_argument("--batch-tokens", type=int, default=65536)
    e.add_argument("--chunk-texts", type=int, default=200_000)
    e.add_argument("--force", action="store_true")

    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("dataset")

    j = sub.add_parser("extscore-synth")
    j.add_argument("dataset")
    j.add_argument("--tag", default="synthetic")
    j.add_argument("--signal", type=float, default=1.5)
    j.add_argument("--coverage", type=float, default=0.9)
    j.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.dataset == "twibot22":
            from .build import twibot22 as t
            steps = {"users": t.build_users, "graph": t.build_graph, "tweets": t.build_tweets}
            if a.step not in steps:
                ap.error(f"--step must be one of {list(steps)}")
            steps[a.step](force=a.force)
        elif a.dataset == "twibot20":
            from .build.twibot20 import build
            build(force=a.force)
        elif a.dataset == "cresci15":
            from .build.cresci15 import build
            build(force=a.force)
    elif a.cmd == "embed":
        from .build.embed import run_embed
        run_embed(a.dataset, encoder=a.encoder, target=a.target, selection=a.selection,
                  max_tokens=a.max_tokens, batch_tokens=a.batch_tokens, chunk_texts=a.chunk_texts,
                  force=a.force)
    elif a.cmd == "status":
        from .status import all_status, format_status
        st = all_status()
        if a.json:
            print(json.dumps(st, indent=2))
        else:
            for s_ in st:
                print(format_status(s_))
    elif a.cmd == "verify":
        from .verify import verify
        ok = verify(a.dataset)
        return 0 if ok else 1
    elif a.cmd == "extscore-synth":
        from .extscore import make_synthetic_extscore
        p = make_synthetic_extscore(a.dataset, tag=a.tag, signal=a.signal, coverage=a.coverage, seed=a.seed)
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
