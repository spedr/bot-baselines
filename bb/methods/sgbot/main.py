"""SGBot (BotometerLite features) random forest (benchmark feature vector by default).

    uv run python -m bb.methods.sgbot --dataset twibot20 --seeds 0 1 2 3 4
    uv run python -m bb.methods.sgbot --dataset twibot22 --with-extscore synthetic
    uv run python -m bb.methods.sgbot --dataset twibot20 --features paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bb.paths import dataset_paths

from ..bothunter.shared import common_args, run_rf_method
from .features import build_features

HERE = Path(__file__).parent


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bb.methods.sgbot", description=__doc__.splitlines()[0])
    common_args(ap)
    ap.add_argument("--features", choices=["benchmark", "paper"], default="benchmark",
                    help="benchmark = TwiBot-22 re-implementation (published numbers); paper = restore "
                         "favourites/default_profile/background-image where present, ratio / max(friends,1)")
    ap.add_argument("--likelihood-corpus", choices=["all_users", "train_users"], default="all_users",
                    help="screen names for the bigram table (all_users = benchmark, label-free transductive)")
    return ap


def variant(a: argparse.Namespace) -> str:
    v = a.features
    if a.likelihood_corpus != "all_users":
        v += "+" + a.likelihood_corpus
    if a.reference_year:
        v += f"+ref{a.reference_year}"
    if a.drop_constant:
        v += "+dropconst"
    if a.split != "official":
        v += f"+{a.split}split{a.split_seed}"
    return v


def main(argv=None) -> list[dict]:
    args = parser().parse_args(argv)
    args.variant = variant(args)
    src = sorted(HERE.glob("*.py")) + [HERE.parent / "bothunter" / "shared.py"]
    return run_rf_method("sgbot", args, build_features, src, [dataset_paths(args.dataset).profile])


if __name__ == "__main__":
    main()
