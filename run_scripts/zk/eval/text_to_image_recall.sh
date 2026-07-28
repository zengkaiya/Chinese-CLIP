#!/bin/bash

dataset_name=MUGE
split=valid # 指定计算valid或test集特征
DATAPATH=evaluation

python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_output_t2i.json
    
cat ${DATAPATH}/datasets/${dataset_name}/${split}_output_t2i.json
