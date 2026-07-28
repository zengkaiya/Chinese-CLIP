#!/usr/bin/env python3
"""Select the best caption candidate for each Tianwang image.

The input is a JSONL file with an image path and ``caption_candidates``.  The
output keeps every input field, replaces ``caption`` with the candidate having
the highest image-text cosine similarity, and adds ``caption_similarity``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from cn_clip.clip.model import CLIP, convert_weights
from cn_clip.clip.utils import image_transform, tokenize
from cn_clip.training.main import convert_models_to_fp32

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter caption candidates by Chinese-CLIP image-text similarity."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--image-root",
        required=True,
        help="Prefix prepended to each record path.",
    )
    parser.add_argument("--resume", required=True, help="Chinese-CLIP checkpoint.")
    parser.add_argument("--vision-model", default="ViT-B-16")
    parser.add_argument("--text-model", default="RoBERTa-wwm-ext-base-chinese")
    parser.add_argument("--context-length", type=int, default=52)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=8,
        help="CPU threads per GPU process for image loading and preprocessing.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument(
        "--skip-invalid-images",
        action="store_true",
        help="Skip records whose image cannot be fully decoded.",
    )
    return parser.parse_args()


def load_model(args):
    config_dir = Path(__file__).resolve().parents[2] / "cn_clip" / "clip" / "model_configs"
    vision_config_path = config_dir / f"{args.vision_model.replace('/', '-')}.json"
    text_config_path = config_dir / f"{args.text_model.replace('/', '-')}.json"

    with vision_config_path.open("r", encoding="utf-8") as vision_file:
        model_info = json.load(vision_file)
    if isinstance(model_info["vision_layers"], str):
        model_info["vision_layers"] = eval(model_info["vision_layers"])
    with text_config_path.open("r", encoding="utf-8") as text_file:
        model_info.update(json.load(text_file))

    model = CLIP(**model_info)
    convert_weights(model)
    convert_models_to_fp32(model)

    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {
            key[len("module.") :]: value
            for key, value in state_dict.items()
            if "bert.pooler" not in key
        }
    model.load_state_dict(state_dict)
    model.cuda(args.gpu)
    model.eval()
    return model, model_info["image_resolution"]


def iter_jsonl(path: Path, start_line: int, end_line: Optional[int]):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file):
            if line_number < start_line:
                continue
            if end_line is not None and line_number >= end_line:
                break
            yield json.loads(line)


def get_candidates(record):
    candidates = []
    seen = set()
    for candidate in record.get("caption_candidates", []):
        if not isinstance(candidate, dict):
            continue
        text = candidate.get("text")
        if not isinstance(text, str) or not text.strip() or text in seen:
            continue
        seen.add(text)
        candidates.append((text, candidate.get("source", "unknown")))

    if not candidates:
        caption = record.get("caption")
        if isinstance(caption, str) and caption.strip():
            candidates.append((caption, record.get("caption_source", "original")))
    return candidates


def process_batch(records, model, preprocess, args, image_pool):
    valid_records = []
    images = []
    skipped = 0
    image_root = Path(args.image_root)

    def load_image(record):
        image_path = image_root / record["path"]
        try:
            image = Image.open(image_path)
            image.load()
            image_tensor = preprocess(image)
            image.close()
            return record, image_tensor, None
        except (OSError, UnidentifiedImageError, KeyError) as exc:
            if not args.skip_invalid_images:
                raise OSError(f"Invalid image {image_path}: {exc}") from exc
            return None, None, str(image_path)

    for record, image_tensor, _ in image_pool.map(load_image, records):
        if record is None:
            skipped += 1
            continue
        valid_records.append(record)
        images.append(image_tensor)

    if not valid_records:
        return [], skipped

    image_tensor = torch.stack(images).pin_memory().cuda(args.gpu, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    candidate_groups = [get_candidates(record) for record in valid_records]
    flat_candidates = []
    group_offsets = [0]
    for candidates in candidate_groups:
        flat_candidates.extend(candidates)
        group_offsets.append(len(flat_candidates))

    if not flat_candidates:
        for record in valid_records:
            record["caption_similarity"] = None
        return valid_records, skipped

    flat_scores = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        for start in range(0, len(flat_candidates), args.text_batch_size):
            texts = [text for text, _ in flat_candidates[start : start + args.text_batch_size]]
            text_tensor = tokenize(texts, context_length=args.context_length).pin_memory().cuda(
                args.gpu, non_blocking=True
            )
            text_features = model.encode_text(text_tensor)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Candidate text features are scored against all images in this
            # batch; retain only the diagonal/group-relevant scores below.
            flat_scores.append(image_features @ text_features.t())

        scores = torch.cat(flat_scores, dim=1).cpu()

    for row_index, record in enumerate(valid_records):
        start, end = group_offsets[row_index], group_offsets[row_index + 1]
        if start == end:
            record["caption_similarity"] = None
            continue
        best_relative_index = scores[row_index, start:end].argmax().item()
        best_index = start + best_relative_index
        best_text, best_source = flat_candidates[best_index]
        record["caption"] = best_text
        record["caption_source"] = best_source
        record["caption_similarity"] = float(scores[row_index, best_index])

    return valid_records, skipped


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for similarity filtering.")
    if args.image_batch_size < 1 or args.text_batch_size < 1:
        raise ValueError("image-batch-size and text-batch-size must be positive")
    if args.image_workers < 1:
        raise ValueError("image-workers must be positive")

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    model, image_resolution = load_model(args)
    preprocess = image_transform(image_resolution)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")

    records_seen = 0
    records_written = 0
    records_skipped = 0
    batch = []
    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            with ThreadPoolExecutor(max_workers=args.image_workers) as image_pool:
                for record in tqdm(
                    iter_jsonl(input_path, args.start_line, args.end_line),
                    desc=f"GPU {args.gpu}",
                    unit="record",
                ):
                    batch.append(record)
                    records_seen += 1
                    if len(batch) < args.image_batch_size:
                        continue

                    processed, skipped = process_batch(batch, model, preprocess, args, image_pool)
                    records_skipped += skipped
                    for processed_record in processed:
                        output_file.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                    records_written += len(processed)
                    batch = []

                if batch:
                    processed, skipped = process_batch(batch, model, preprocess, args, image_pool)
                    records_skipped += skipped
                    for processed_record in processed:
                        output_file.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                    records_written += len(processed)

        os.replace(temporary_output, output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    print(
        f"GPU {args.gpu}: seen={records_seen}, written={records_written}, "
        f"skipped={records_skipped}, output={output_path}"
    )


if __name__ == "__main__":
    run(parse_args())
