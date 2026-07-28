#!/bin/bash

dataset_name=MUGE
split=valid # 指定计算valid或test集特征
DATAPATH=evaluation

python cn_clip/eval/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

python cn_clip/eval/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_output_i2t.json

cat ${DATAPATH}/datasets/${dataset_name}/${split}_output_i2t.json