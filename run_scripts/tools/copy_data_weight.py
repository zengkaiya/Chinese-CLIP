#!/usr/bin/env python3

"""Copy the latest checkpoint from Tianwang experiments."""

import argparse
import shutil
from pathlib import Path


def copy_latest_checkpoints(experiments_dir: Path, output_dir: Path) -> int:
    """Copy epoch_latest.pt for every tianwang experiment found."""
    copied = 0
    for experiment_dir in sorted(experiments_dir.iterdir()):
        if not experiment_dir.is_dir() or not experiment_dir.name.startswith("tianwang"):
            continue

        source = experiment_dir / "checkpoints" / "epoch_latest.pt"
        if not source.is_file():
            print(f"skip {experiment_dir.name}: {source} does not exist")
            continue

        destination = output_dir / experiment_dir.name / "checkpoints" / "epoch_latest.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {source} -> {destination}")
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy epoch_latest.pt from tianwang experiments."
    )
    parser.add_argument(
        "experiments_dir",
        nargs="?",
        type=Path,
        default=Path("evaluation/experiments"),
        help="directory containing experiment directories (default: evaluation/experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/tianwang_exp/ckpts"),
        help="destination directory (default: evaluation/tianwang_exp/ckpts)",
    )
    args = parser.parse_args()

    if not args.experiments_dir.is_dir():
        parser.error(f"experiments directory does not exist: {args.experiments_dir}")

    copied = copy_latest_checkpoints(args.experiments_dir, args.output_dir)
    print(f"copied {copied} checkpoint(s)")


if __name__ == "__main__":
    main()
