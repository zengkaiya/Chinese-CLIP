#!/usr/bin/env python3
"""Collect zeroshot-top1 results from a multi-dataset evaluation log."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


DATASET_PATTERN = re.compile(r"^=====\s*(.*?)\s*=====\s*$")
RESULT_PATTERN = re.compile(r"zeroshot-top1\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract dataset-to-zeroshot-top1 mappings from an evaluation log."
    )
    parser.add_argument(
        "--input",
        default="evaluation/zeroshot_classify/coco-cn_finetune_vit-b-16/zeroshot_classify_multi_results.txt",
        help="Evaluation log path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Optional number of decimal places. By default, preserve the logged value.",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "zeroshot", "retrieval"],
        default="auto",
        help="Log format. Auto-detect by default.",
    )
    return parser.parse_args()


def collect_results(input_path: Path):
    results = {}
    current_dataset = None
    duplicate_datasets = set()

    with input_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            dataset_match = DATASET_PATTERN.match(line.strip())
            if dataset_match:
                current_dataset = dataset_match.group(1)
                continue

            result_match = RESULT_PATTERN.search(line)
            if result_match and current_dataset:
                if current_dataset in results:
                    duplicate_datasets.add(current_dataset)
                results[current_dataset] = result_match.group(1)

    return results, duplicate_datasets


def format_results(results, digits=None):
    lines = []
    for dataset, raw_score in results.items():
        if digits is None:
            score = raw_score
        else:
            score = f"{float(raw_score):.{digits}f}"
        lines.append(f"{dataset}\t{score}")
    return "\n".join(lines) + ("\n" if lines else "")


def collect_retrieval_results(input_path: Path):
    results = []
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as log_file:
        reader = csv.DictReader(log_file, delimiter="\t")
        required = {"dataset", "split", "direction", "result"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("retrieval log must contain dataset, split, direction, result columns")

        for line_number, row in enumerate(reader, start=2):
            if not row.get("result") or (row.get("dataset") or "").startswith("#"):
                continue
            try:
                result = json.loads(row["result"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid result JSON: {exc}") from exc

            score_json = result.get("scoreJson", {})
            results.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "direction": row["direction"],
                    "success": result.get("success", False),
                    "score": score_json.get("score", result.get("score")),
                    "r1": score_json.get("r1"),
                    "r5": score_json.get("r5"),
                    "r10": score_json.get("r10"),
                }
            )
    return results


def format_retrieval_results(results, digits=None):
    headers = ["dataset", "split", "direction", "success", "score", "r1", "r5", "r10"]
    lines = ["\t".join(headers)]
    for result in results:
        values = []
        for key in headers:
            value = result[key]
            if isinstance(value, (int, float)) and digits is not None:
                value = f"{value:.{digits}f}"
            elif value is None:
                value = ""
            values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines) + ("\n" if lines else "")


def detect_task(input_path: Path):
    with input_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if line.strip():
                return "retrieval" if line.startswith("dataset\tsplit\tdirection\tresult") else "zeroshot"
    return "zeroshot"


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input log does not exist: {input_path}", file=sys.stderr)
        return 1

    task = detect_task(input_path) if args.task == "auto" else args.task
    if task == "retrieval":
        retrieval_results = collect_retrieval_results(input_path)
        output = format_retrieval_results(retrieval_results, args.digits)
        result_count = len(retrieval_results)
        duplicate_datasets = set()
    else:
        results, duplicate_datasets = collect_results(input_path)
        output = format_results(results, args.digits)
        result_count = len(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"Collected {result_count} results to {output_path}")
    else:
        sys.stdout.write(output)

    if duplicate_datasets:
        print(
            "Warning: duplicate datasets found; the last result was kept: "
            + ", ".join(sorted(duplicate_datasets)),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
