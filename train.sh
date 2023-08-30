#!/bin/sh
python train_CLIP/train.py \
        --model_name ViT-B/16 \
        --folder data \
	--csv data/polyvore_img_desc.csv \
        --batch_size 4 \
        --accelerator gpu \
        --num_workers 8 \
        --max_steps 45000 \
        --shuffle True 
