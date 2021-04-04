#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES=0,1,2 /HDD/ningbo/anaconda3/envs/mmlab/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --gpu-ids 1 2 --launcher pytorch ${@:3}  
