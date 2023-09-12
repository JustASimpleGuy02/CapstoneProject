#!/bin/sh

# Declare an array of string with type
# declare -a SamplePrompts=(
#     "outfit to go to the street on sunday night",
#     "ski outfit"
# )

python apis/search_item_fashion_clip.py \
       --image-dir data/hm/fashion_items_test \
       --prompt "ski outfit" \
       --embeddings-file model_embeddings/fashion-clip/image_embeddings_demo.txt \
       -k 10 

