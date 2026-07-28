#!/bin/bash

cd /mnt/bn/yuyingchen/zk/project/miscs/Chinese-CLIP

python run_scripts/tools/convert_tianwang_manifest.py \
    --manifest /mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/pairs/2015/image_text_pairs_new_with_caption_similarity_filtered.jsonl \
    --tianwang-root /mnt/bn/yuyingchen/zk/project/miscs/tianwang \
    --output-dir evaluation/datasets/tianwang_3M \
    --workers 128 \
    --skip-invalid-images \
    --skip-empty-captions

python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir evaluation/datasets/tianwang_3M \
    --splits train