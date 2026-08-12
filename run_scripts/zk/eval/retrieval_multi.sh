#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_root}"

# Only supports single-GPU inference.
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/cn_clip"

dataset_name=tianwang51w

DATAPATH=evaluation
resume=evaluation/experiments/${dataset_name}_finetune_vit-b-16_roberta-base/checkpoints/epoch_latest.pt
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
result_file=${DATAPATH}/retrieval/${dataset_name}_finetune_vit-b-16.txt

# Each entry is: dataset name, split.
datasets=(
    "MUGE valid"
    "Flickr30k-CN test"
    "COCO-CN test"
)

run_dataset() {
    local dataset_name="$1"
    local split="$2"
    local dataset_dir="${DATAPATH}/datasets/${dataset_name}"
    local image_feats="${dataset_dir}/${split}_imgs.img_feat.jsonl"
    local text_feats="${dataset_dir}/${split}_texts.txt_feat.jsonl"
    local t2i_predictions="${dataset_dir}/${split}_predictions.jsonl"
    local i2t_predictions="${dataset_dir}/${split}_tr_predictions.jsonl"
    local t2i_result="${dataset_dir}/${split}_output_t2i.json"
    local i2t_result="${dataset_dir}/${split}_output_i2t.json"
    local tr_texts="${dataset_dir}/${split}_texts.tr.jsonl"

    echo
    echo "===== ${dataset_name} ${split}: extracting features ====="
    python -u cn_clip/eval/extract_features.py \
        --extract-image-feats \
        --extract-text-feats \
        --image-data="${dataset_dir}/lmdb/${split}/imgs" \
        --text-data="${dataset_dir}/${split}_texts.jsonl" \
        --img-batch-size=32 \
        --text-batch-size=32 \
        --context-length=52 \
        --resume="${resume}" \
        --vision-model="${vision_model}" \
        --text-model="${text_model}"

    echo "===== ${dataset_name} ${split}: text-to-image retrieval ====="
    python -u cn_clip/eval/make_topk_predictions.py \
        --image-feats="${image_feats}" \
        --text-feats="${text_feats}" \
        --top-k=10 \
        --eval-batch-size=32768 \
        --output="${t2i_predictions}"

    python -u cn_clip/eval/evaluation.py \
        "${dataset_dir}/${split}_texts.jsonl" \
        "${t2i_predictions}" \
        "${t2i_result}"
    printf '%s\t%s\ttext-to-image\t%s\n' \
        "${dataset_name}" "${split}" "$(<"${t2i_result}")" | tee -a "${result_file}"

    echo "===== ${dataset_name} ${split}: image-to-text retrieval ====="
    python -u cn_clip/eval/make_topk_predictions_tr.py \
        --image-feats="${image_feats}" \
        --text-feats="${text_feats}" \
        --top-k=10 \
        --eval-batch-size=32768 \
        --output="${i2t_predictions}"

    python -u cn_clip/eval/transform_ir_annotation_to_tr.py \
        --input "${dataset_dir}/${split}_texts.jsonl"

    python -u cn_clip/eval/evaluation_tr.py \
        "${tr_texts}" \
        "${i2t_predictions}" \
        "${i2t_result}"
    printf '%s\t%s\timage-to-text\t%s\n' \
        "${dataset_name}" "${split}" "$(<"${i2t_result}")" | tee -a "${result_file}"
}

mkdir -p "${DATAPATH}"
{
    echo -e "dataset\tsplit\tdirection\tresult"
    echo "# Retrieval evaluation results"
} > "${result_file}"

for dataset_spec in "${datasets[@]}"; do
    read -r dataset_name split <<< "${dataset_spec}"
    run_dataset "${dataset_name}" "${split}"
done

echo
echo "All retrieval evaluations finished. Results saved to: ${result_file}"
