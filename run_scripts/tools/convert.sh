#!/bin/bash

cd /mnt/bn/yuyingchen/zk/project/miscs/Chinese-CLIP

# python run_scripts/tools/convert_tianwang_manifest.py \
#     --manifest /mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/pairs/2015/image_text_pairs_new_with_caption_similarity_filtered.jsonl \
#     --tianwang-root /mnt/bn/yuyingchen/zk/project/miscs/tianwang \
#     --output-dir evaluation/datasets/tianwang_3M \
#     --workers 128 \
#     --skip-invalid-images \
#     --skip-empty-captions

# python run_scripts/tools/convert_tianwang_recordio.py \
#     --input-prefix /mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/danqing/2015_v3/final \
#     --output-dir evaluation/datasets/tianwang50w \
#     --tokenizer-src /mnt/bn/yuyingchen/zk/project/miscs/tianwang/cn-clip-ctp/src \
#     --workers 64 \
#     --skip-invalid-images

python run_scripts/tools/convert_tianwang_recordio_multi.py \
    --input-prefix /mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/final/danqing/2015_v3/final,/mnt/bn/yuyingchen/zk/project/miscs/tianwang/outputs/2013_2014/danqing/final \
    --output-dir evaluation/datasets/tianwang51w \
    --tokenizer-src /mnt/bn/yuyingchen/zk/project/miscs/tianwang/cn-clip-ctp/src \
    --workers 64 \
    --skip-invalid-images

python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir evaluation/datasets/tianwang51w \
    --splits train