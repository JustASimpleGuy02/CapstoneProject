#!/bin/sh
python apis/search_item.py \
       --image-dir data/fashion_items_test \
       --model-name RN50 \
       --model-path ../training_logs/2023_08_21/epoch=47-step=43248.ckpt \
       --prompt "blue shirt" \
       --embeddings-file model_embeddings/2023_08_21/image_embeddings_demo.txt \
       -k 5 

