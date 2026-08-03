#!/usr/bin/env python3
"""Convert Tianwang MXNet RecordIO directly to Chinese-CLIP train files.

The RecordIO header contains BERT token IDs in ``header.label`` and the
record payload contains encoded image bytes.  This tool writes:

    train_imgs.tsv       image_id<TAB>base64 image bytes
    train_texts.jsonl    {"text_id": ..., "text": ..., "image_ids": [...]}

It does not materialize decoded image files on disk.
"""

import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
import sys
from pathlib import Path

import mxnet as mx
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm


DEFAULT_INPUT_PREFIX = (
    "/mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/2013_2014/"
    "danqing/final"
)
DEFAULT_OUTPUT_DIR = "evaluation/datasets/tianwang"
DEFAULT_TOKENIZER_SRC = (
    "/mnt/bn/yuyingchen/zk/project/miscs/tianwang/cn-clip-ctp/src"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MXNet RecordIO directly to Chinese-CLIP train files."
    )
    parser.add_argument(
        "--input-prefix",
        default=DEFAULT_INPUT_PREFIX,
        help=(
            "Comma-separated RecordIO prefixes without .idx/.rec; spaces after "
            "commas are allowed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train_imgs.tsv and train_texts.jsonl.",
    )
    parser.add_argument(
        "--tokenizer-src",
        default=DEFAULT_TOKENIZER_SRC,
        help="Directory containing open_clip.bert_tokenizer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="Number of threads used for image validation and encoding.",
    )
    parser.add_argument(
        "--skip-invalid-images",
        action="store_true",
        help="Skip records whose image cannot be fully decoded by PIL.",
    )
    return parser.parse_args()


def load_bert_tokenizer(tokenizer_src: Path):
    sys.path.insert(0, str(tokenizer_src))
    try:
        from open_clip.bert_tokenizer import _tokenizer
    except ImportError as exc:
        raise ImportError(
            f"Cannot import open_clip.bert_tokenizer from {tokenizer_src}"
        ) from exc
    return _tokenizer


def decode_text(label, tokenizer):
    token_ids = [int(token_id) for token_id in label[:52] if int(token_id) != 0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return (
        "".join(tokens)
        .replace("[CLS]", "")
        .replace("[SEP]", "")
        .replace("[PAD]", "")
        .replace("[UNK]", "")
        .replace("#", "")
        .strip()
    )


def encode_record(item, tokenizer, skip_invalid_images):
    output_index, record_key, packed = item
    header, image_bytes = mx.recordio.unpack(packed)
    caption = decode_text(header.label, tokenizer)
    if not caption:
        raise ValueError(f"record {record_key} has an empty decoded caption")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        message = f"record {record_key}: invalid image: {exc}"
        if skip_invalid_images:
            return output_index, None, message
        raise OSError(message) from exc

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_id = output_index
    text_id = output_index
    return output_index, text_id, image_id, caption, image_b64


def parse_input_prefixes(input_prefixes: str):
    prefixes = [
        Path(prefix.strip())
        for prefix in input_prefixes.split(",")
        if prefix.strip()
    ]
    if not prefixes:
        raise ValueError("input-prefix must contain at least one RecordIO prefix")
    return prefixes


def convert(
    input_prefixes: str,
    output_dir: Path,
    tokenizer_src: Path,
    workers: int,
    skip_invalid_images: bool,
):
    prefixes = parse_input_prefixes(input_prefixes)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    for input_prefix in prefixes:
        idx_path = input_prefix.with_suffix(".idx")
        rec_path = input_prefix.with_suffix(".rec")
        if not idx_path.is_file():
            raise FileNotFoundError(f"RecordIO index does not exist: {idx_path}")
        if not rec_path.is_file():
            raise FileNotFoundError(f"RecordIO data does not exist: {rec_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_tmp = output_dir / "train_imgs.tsv.tmp"
    text_tmp = output_dir / "train_texts.jsonl.tmp"
    image_output = output_dir / "train_imgs.tsv"
    text_output = output_dir / "train_texts.jsonl"
    tokenizer = load_bert_tokenizer(tokenizer_src)
    readers = []
    try:
        for prefix in prefixes:
            readers.append(
                mx.recordio.MXIndexedRecordIO(
                    str(prefix.with_suffix(".idx")),
                    str(prefix.with_suffix(".rec")),
                    "r",
                )
            )
    except BaseException:
        for reader in readers:
            reader.close()
        raise
    total_records = sum(len(reader.keys) for reader in readers)
    records = 0
    skipped = 0
    skipped_messages = []

    try:
        with image_tmp.open("w", encoding="ascii", newline="\n") as image_file:
            with text_tmp.open("w", encoding="utf-8", newline="\n") as text_file:
                pending = deque()
                window_size = max(workers * 4, workers)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    def record_iter():
                        output_index = 1
                        for source_index, reader in enumerate(readers):
                            for record_key in reader.keys:
                                packed = reader.read_idx(record_key)
                                yield output_index, source_index, record_key, packed
                                output_index += 1

                    key_iter = iter(record_iter())

                    def submit_next():
                        output_index, source_index, record_key, packed = next(key_iter)
                        return executor.submit(
                            encode_record,
                            (
                                output_index,
                                f"input {source_index}, record {record_key}",
                                packed,
                            ),
                            tokenizer,
                            skip_invalid_images,
                        )

                    for _ in range(window_size):
                        try:
                            pending.append(submit_next())
                        except StopIteration:
                            break

                    with tqdm(total=total_records, desc="Converting RecordIO", unit="record") as progress:
                        while pending:
                            result = pending.popleft().result()
                            if result[1] is None:
                                skipped += 1
                                if len(skipped_messages) < 20:
                                    skipped_messages.append(result[2])
                            else:
                                _, text_id, image_id, caption, image_b64 = result
                                image_file.write(f"{image_id}\t{image_b64}\n")
                                text_file.write(
                                    json.dumps(
                                        {
                                            "text_id": text_id,
                                            "text": caption,
                                            "image_ids": [image_id],
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                records += 1
                            progress.update(1)

                            try:
                                pending.append(submit_next())
                            except StopIteration:
                                pass
        os.replace(image_tmp, image_output)
        os.replace(text_tmp, text_output)
    except BaseException:
        image_tmp.unlink(missing_ok=True)
        text_tmp.unlink(missing_ok=True)
        raise
    finally:
        for reader in readers:
            reader.close()

    print(f"Converted {records} records.")
    print(f"Skipped invalid images: {skipped}")
    for message in skipped_messages:
        print(f"  {message}", file=sys.stderr)
    print(f"Images: {image_output}")
    print(f"Texts:  {text_output}")


def main():
    args = parse_args()
    try:
        convert(
            args.input_prefix,
            Path(args.output_dir),
            Path(args.tokenizer_src),
            args.workers,
            args.skip_invalid_images,
        )
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
