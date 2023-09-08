#!/bin/sh
python apis/search_item_fashion-clip.py \
       --image-dir data/hm/fashion_items_test \
       --prompt "modern outfit to go to the street on sunday night" \
       --embeddings-file model_embeddings/fashion-clip/image_embeddings_demo.txt \
       -k 10 

