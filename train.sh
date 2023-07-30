#!/bin/sh
python train_CLIP/train.py \
        --model_name RN50 \
        --folder data \
        --batch_size 16 \
        --accelerator gpu \
        --num_workers 8 \
        --max_steps 10000 
