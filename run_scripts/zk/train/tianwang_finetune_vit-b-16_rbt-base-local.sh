#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

cd /mnt/bn/yuyingchen/zk/project/miscs/Chinese-CLIP
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

export NCCL_SOCKET_FAMILY=AF_INET   # 强制使用 IPv4，避免 IPv6 干扰
unset NCCL_NET_PLUGIN                # 禁用 FasTrak 插件自动注入
export NCCL_IB_DISABLE=1             # 若无 IB/RDMA，禁用以减少不确定性

# Number of GPUs per GPU worker
GPUS_PER_NODE=8
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

DATAPATH=evaluation
dataset_name=tianwang36w

# data options
train_data=${DATAPATH}/datasets/${dataset_name}/lmdb/train
val_data=${DATAPATH}/datasets/COCO-CN/lmdb/valid # if val_data is not specif  ied, the validation will be automatically disabled
# val_data=""

# restore options
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt # or specify your customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=${DATAPATH}/experiments/
name=${dataset_name}_finetune_vit-b-16_roberta-base
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=52
warmup=6
batch_size=1024
valid_batch_size=128
accum_freq=1
lr=3e-5
wd=0.001
max_epochs=10
valid_step_interval=999999
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"
# use_augment=""

python -m torch.distributed.run \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${WORKER_CNT} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    cn_clip/training/main.py \
    --train-data=${train_data} \
    --val-data=${val_data} \
    --resume=${resume} \
    ${reset_data_offset} \
    ${reset_optimizer} \
    --output-dir=${output_base_dir} \
    --name=${name} \
    --save-step-frequency=${save_step_frequency} \
    --save-epoch-frequency=${save_epoch_frequency} \
    --log-interval=${log_interval} \
    ${report_training_batch_acc} \
    --context-length=${context_length} \
    --warmup=${warmup} \
    --batch-size=${batch_size} \
    --valid-batch-size=${valid_batch_size} \
    --valid-step-interval=${valid_step_interval} \
    --valid-epoch-interval=${valid_epoch_interval} \
    --accum-freq=${accum_freq} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    ${use_augment} \
    --text-model=${text_model} \
    --grad-checkpointing