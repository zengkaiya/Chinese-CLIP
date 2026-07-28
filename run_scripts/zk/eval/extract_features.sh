#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

dataset_name=MUGE
split=valid # 指定计算valid或test集特征
DATAPATH=evaluation
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese

python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model}
