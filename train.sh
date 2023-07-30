#!/bin/sh
python train_CLIP/train.py \
        --model_name RN50 \
        --folder data_dir \
        --batch_size 2 \
        --gpus 0
