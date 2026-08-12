#!/bin/bash

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
tianwang_root=${TIANWANG_ROOT:-../tianwang}

# python run_scripts/tools/convert_tianwang_manifest.py \
#     --manifest "${tianwang_root}/outputs/final/pairs/2015/image_text_pairs_new_with_caption_similarity_filtered.jsonl" \
#     --tianwang-root "${tianwang_root}" \
#     --output-dir evaluation/datasets/tianwang_3M \
#     --workers 128 \
#     --skip-invalid-images \
#     --skip-empty-captions

# python run_scripts/tools/convert_tianwang_recordio.py \
#     --input-prefix "${tianwang_root}/outputs/final/danqing/2015_v3/final" \
#     --output-dir evaluation/datasets/tianwang50w \
#     --tokenizer-src "${tianwang_root}/cn-clip-ctp/src" \
#     --workers 64 \
#     --skip-invalid-images

python run_scripts/tools/convert_tianwang_recordio_multi.py \
    --input-prefix "${tianwang_root}/outputs/final/danqing/2015_v3/final,${tianwang_root}/outputs/2013_2014/danqing/final" \
    --output-dir evaluation/datasets/tianwang51w \
    --tokenizer-src "${tianwang_root}/cn-clip-ctp/src" \
    --workers 64 \
    --skip-invalid-images

python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir evaluation/datasets/tianwang51w \
    --splits train