#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/cn_clip"

input_jsonl=/mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/pairs/2015/image_text_pairs_new_with_caption.jsonl
image_root=/mnt/bn/yuyingchen/zk/project/miscs/tianwang
resume=evaluation/pretrained_weights/clip_cn_vit-b-16.pt
output_jsonl=/mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/pairs/2015/image_text_pairs_new_with_caption_similarity_filtered.jsonl

# Set GPU_IDS to the physical GPUs to use, for example: GPU_IDS=0,1,2,3
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
image_batch_size=${IMAGE_BATCH_SIZE:-256}
text_batch_size=${TEXT_BATCH_SIZE:-512}
image_workers=${IMAGE_WORKERS:-32}

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
gpu_count=${#gpu_ids[@]}
if [[ ${gpu_count} -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU" >&2
    exit 1
fi
if [[ ! -f "${input_jsonl}" ]]; then
    echo "Input JSONL does not exist: ${input_jsonl}" >&2
    exit 1
fi

output_dir=$(dirname "${output_jsonl}")
mkdir -p "${output_dir}"
tmp_dir=$(mktemp -d "${output_dir}/.caption_similarity.XXXXXX")

cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

# Split once by line count. Each GPU then reads only its own shard instead of
# repeatedly scanning the prefix of the original multi-million-line file.
echo "Splitting input into ${gpu_count} line-balanced shards..."
split -n "l/${gpu_count}" -d -a 4 "${input_jsonl}" "${tmp_dir}/input_"

pids=()
for ((shard_id = 0; shard_id < gpu_count; shard_id++)); do
    shard_input=${tmp_dir}/input_$(printf '%04d' "${shard_id}")
    shard_output=${tmp_dir}/shard_${shard_id}.jsonl
    shard_log=${tmp_dir}/shard_${shard_id}.log

    # GNU split may omit empty trailing shards when there are fewer records
    # than requested shards.
    if [[ ! -f "${shard_input}" ]]; then
        : > "${shard_input}"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard_id]}" python -u \
        run_scripts/tools/filter_tianwang_caption_similarity.py \
        --input-jsonl "${shard_input}" \
        --image-root "${image_root}" \
        --output-jsonl "${shard_output}" \
        --resume "${resume}" \
        --gpu 0 \
        --image-batch-size "${image_batch_size}" \
        --text-batch-size "${text_batch_size}" \
        --image-workers "${image_workers}" \
        --skip-invalid-images >"${shard_log}" 2>&1 &
    pids+=("$!")
done

failed=0
for shard_id in "${!pids[@]}"; do
    if ! wait "${pids[$shard_id]}"; then
        failed=1
    fi
    echo "===== shard ${shard_id} log ====="
    cat "${tmp_dir}/shard_${shard_id}.log"
done
if [[ ${failed} -ne 0 ]]; then
    echo "One or more similarity filtering workers failed." >&2
    exit 1
fi

temporary_output=${output_jsonl}.tmp
: > "${temporary_output}"
for ((shard_id = 0; shard_id < gpu_count; shard_id++)); do
    cat "${tmp_dir}/shard_${shard_id}.jsonl" >> "${temporary_output}"
done
mv "${temporary_output}" "${output_jsonl}"
echo "Merged result: ${output_jsonl}"
