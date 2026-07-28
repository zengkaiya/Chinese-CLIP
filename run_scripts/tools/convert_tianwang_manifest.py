#!/usr/bin/env python3
"""Convert the Tianwang image-text manifest to Chinese-CLIP training files.

The input manifest contains one image-text pair per JSONL line.  The output
format follows README.md:

    train_imgs.tsv       image_id<TAB>base64 image bytes
    train_texts.jsonl    {"text_id": ..., "text": ..., "image_ids": [...]}
"""

import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm


DEFAULT_MANIFEST = (
    "/mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/2013_2014/"
    "images/image_text_pairs_recordio_manifest.jsonl"
)
DEFAULT_TIANWANG_ROOT = "/mnt/bn/yuyingchen/zk/project/miscs/tianwang"
DEFAULT_OUTPUT_DIR = "evaluation/datasets/tianwang"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Tianwang manifest records to Chinese-CLIP train files."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Input JSONL manifest.")
    parser.add_argument(
        "--tianwang-root",
        default=DEFAULT_TIANWANG_ROOT,
        help="Root directory prepended to each relative local_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train_imgs.tsv and train_texts.jsonl.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Number of threads used for image reading and base64 encoding.",
    )
    parser.add_argument(
        "--skip-invalid-images",
        action="store_true",
        help="Skip images that PIL cannot fully decode instead of aborting conversion.",
    )
    parser.add_argument(
        "--skip-empty-captions",
        action="store_true",
        help="Skip records whose caption is empty or only whitespace.",
    )
    return parser.parse_args()


def fail(manifest_path, line_number, message):
    raise ValueError(f"{manifest_path}:{line_number}: {message}")


def count_records(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as manifest:
        return sum(bool(line.strip()) for line in manifest)


def encode_record(
    item,
    tianwang_root: Path,
    skip_invalid_images: bool,
    skip_empty_captions: bool,
):
    line_number, line = item
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc

    if "caption" not in record:
        raise ValueError(f"line {line_number}: missing field 'caption'")

    # Support both the original manifest schema and the newer pair manifest:
    #   old: source_idx, image_id, local_path
    #   new: path, with no explicit IDs; use the 1-based line number as IDs.
    path_value = record.get("local_path", record.get("path"))
    if path_value is None:
        raise ValueError(f"line {line_number}: missing field 'local_path' or 'path'")

    try:
        text_id = int(record.get("source_idx", record.get("text_id", line_number)))
        image_id = int(record.get("image_id", line_number))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: source_idx and image_id must be integers") from exc

    caption = record["caption"]
    if not isinstance(caption, str) or not caption.strip():
        if skip_empty_captions:
            return line_number, None, f"line {line_number}: empty caption"
        raise ValueError(f"line {line_number}: caption must be a non-empty string")

    image_path = tianwang_root / Path(str(path_value))
    if not image_path.is_file():
        raise FileNotFoundError(f"line {line_number}: image does not exist: {image_path}")

    try:
        # Reading the raw bytes alone does not detect truncated JPEG/PNG/TIFF
        # files. Force PIL to decode all pixels before accepting the image.
        with Image.open(image_path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        message = f"line {line_number}: invalid image {image_path}: {exc}"
        if skip_invalid_images:
            return line_number, None, message
        raise OSError(message) from exc

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return line_number, text_id, image_id, caption, image_b64


def convert(
    manifest_path: Path,
    tianwang_root: Path,
    output_dir: Path,
    workers: int,
    skip_invalid_images: bool,
    skip_empty_captions: bool,
):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary files first, then replace the final files together as
    # far as possible, so a failed conversion does not leave half-written data.
    image_tmp = output_dir / "train_imgs.tsv.tmp"
    text_tmp = output_dir / "train_texts.jsonl.tmp"
    image_output = output_dir / "train_imgs.tsv"
    text_output = output_dir / "train_texts.jsonl"

    image_ids = set()
    text_ids = set()
    records = 0
    skipped = 0
    skipped_messages = []
    total_records = count_records(manifest_path)

    try:
        with manifest_path.open("r", encoding="utf-8") as manifest:
            with image_tmp.open("w", encoding="ascii", newline="\n") as image_file:
                with text_tmp.open("w", encoding="utf-8", newline="\n") as text_file:
                    items = ((line_number, line) for line_number, line in enumerate(manifest, 1) if line.strip())
                    pending = deque()
                    window_size = max(workers * 4, workers)

                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        for _ in range(window_size):
                            try:
                                pending.append(
                                    executor.submit(
                                        encode_record,
                                        next(items),
                                        tianwang_root,
                                        skip_invalid_images,
                                        skip_empty_captions,
                                    )
                                )
                            except StopIteration:
                                break

                        with tqdm(total=total_records, desc="Converting", unit="record") as progress:
                            while pending:
                                result = pending.popleft().result()
                                if result[1] is None:
                                    line_number, message = result[0], result[2]
                                    skipped += 1
                                    if len(skipped_messages) < 20:
                                        skipped_messages.append(message)
                                    progress.update(1)
                                    try:
                                        pending.append(
                                            executor.submit(
                                                encode_record,
                                                next(items),
                                                tianwang_root,
                                                skip_invalid_images,
                                                skip_empty_captions,
                                            )
                                        )
                                    except StopIteration:
                                        pass
                                    continue

                                line_number, text_id, image_id, caption, image_b64 = result

                                if image_id in image_ids:
                                    fail(manifest_path, line_number, f"duplicate image_id {image_id}")
                                if text_id in text_ids:
                                    fail(manifest_path, line_number, f"duplicate source_idx/text_id {text_id}")

                                image_file.write(f"{image_id}\t{image_b64}\n")
                                text_file.write(
                                    json.dumps(
                                        {"text_id": text_id, "text": caption, "image_ids": [image_id]},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                image_ids.add(image_id)
                                text_ids.add(text_id)
                                records += 1
                                progress.update(1)

                                try:
                                    pending.append(
                                        executor.submit(
                                            encode_record,
                                            next(items),
                                            tianwang_root,
                                            skip_invalid_images,
                                            skip_empty_captions,
                                        )
                                    )
                                except StopIteration:
                                    pass

        os.replace(image_tmp, image_output)
        os.replace(text_tmp, text_output)
    except BaseException:
        image_tmp.unlink(missing_ok=True)
        text_tmp.unlink(missing_ok=True)
        raise

    print(f"Converted {records} records.")
    print(f"Skipped records: {skipped}")
    for message in skipped_messages:
        print(f"  {message}", file=sys.stderr)
    print(f"Images: {image_output}")
    print(f"Texts:  {text_output}")


def main():
    args = parse_args()
    try:
        convert(
            Path(args.manifest),
            Path(args.tianwang_root),
            Path(args.output_dir),
            args.workers,
            args.skip_invalid_images,
            args.skip_empty_captions,
        )
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
