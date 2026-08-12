#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_root}"

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/cn_clip"

dataset_name=tianwang51w

vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
resume=evaluation/experiments/${dataset_name}_finetune_vit-b-16_roberta-base/checkpoints/epoch_latest.pt
index=""

path=evaluation
savedir=${path}/zeroshot_classify/${dataset_name}_finetune_vit-b-16
result_file=${savedir}/zeroshot_classify_multi_results.txt

mapfile -t datasets < <(
    find "${path}/datasets/all_zip" -maxdepth 1 -type f -name '*.zip' -printf '%f\n' |
        sed 's/\.zip$//' |
        grep -Ev '^(country211|fgvc-aircraft-2013b-variants102|voc-2007-classification)$' |
        sort
)

if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "No ZIP datasets found in ${path}/datasets/all_zip" >&2
    exit 1
fi

mkdir -p "${savedir}"
: > "${result_file}"

echo "Zero-shot evaluation results" | tee -a "${result_file}"
echo "================================" | tee -a "${result_file}"

for dataset in "${datasets[@]}"; do
    datapath=${path}/datasets/${dataset}/test
    label_file=${path}/datasets/${dataset}/label_cn.txt

    echo | tee -a "${result_file}"
    echo "===== ${dataset} =====" | tee -a "${result_file}"

    python -u cn_clip/eval/zeroshot_evaluation.py \
        --datapath="${datapath}" \
        --label-file="${label_file}" \
        --save-dir="${savedir}" \
        --dataset="${dataset}" \
        --index="${index}" \
        --img-batch-size=64 \
        --resume="${resume}" \
        --vision-model="${vision_model}" \
        --text-model="${text_model}" 2>&1 | tee -a "${result_file}"
done

echo | tee -a "${result_file}"
echo "All datasets finished. Results saved to: ${result_file}" | tee -a "${result_file}"
