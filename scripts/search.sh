#!/bin/sh
python apis/search_item.py \
       --image-dir data/fashion_items_test \
       --model-name RN50 \
       --model-path ../training_logs/2023_08_29/epoch=14-step=44000.ckpt \
       --prompt "blue shirt" \
       --embeddings-file model_embeddings/2023_08_29/image_embeddings_demo.txt \
       -k 5 

