"""BotHunter Tier-1 random forest (benchmark feature vector by default).

    uv run python -m bb.methods.bothunter --dataset twibot20 --seeds 0 1 2 3 4
    uv run python -m bb.methods.bothunter --dataset twibot22 --with-extscore synthetic
    uv run python -m bb.methods.bothunter --dataset cresci15 --features extended
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bb.paths import dataset_paths

from .features import build_features
from .shared import common_args, run_rf_method

HERE = Path(__file__).parent


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bb.methods.bothunter", description=__doc__.splitlines()[0])
    common_args(ap)
    ap.add_argument("--features", choices=["benchmark", "extended"], default="benchmark",
                    help="benchmark = 17 slots with 7 constant zeros (published numbers); extended = fill "
                         "favourites + newest-tweet slots where the dataset has them (ablation)")
    ap.add_argument("--tw20-literal-none-location", action="store_true",
                    help="has_location from the raw release spelling (benchmark node file may keep 'None ')")
    ap.add_argument("--bot-ref-casefold", action="store_true",
                    help="match 'bot' case-insensitively (benchmark: case-sensitive)")
    ap.add_argument("--source-topk", type=int, default=10, help="extended: top-k tweet sources kept as codes")
    return ap


def variant(a: argparse.Namespace) -> str:
    v = a.features
    if a.tw20_literal_none_location:
        v += "+nonelocation"
    if a.bot_ref_casefold:
        v += "+casefold"
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
    dp = dataset_paths(args.dataset)
    data = [dp.profile] + ([dp.tweets / "offsets.i64.npy"] if args.features == "extended" else [])
    return run_rf_method("bothunter", args, build_features, sorted(HERE.glob("*.py")), data)


if __name__ == "__main__":
    main()
