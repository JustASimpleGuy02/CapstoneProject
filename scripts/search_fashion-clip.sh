#!/bin/sh
python apis/search_item_fashion-clip.py \
       --image-dir data/hm/fashion_items_test \
       --prompt "ski suit" \
       --embeddings-file model_embeddings/fashion-clip/image_embeddings_demo.txt \
       -k 10 

