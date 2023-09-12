# %%
import os
import os.path as osp
import sys

sys.path += ["../train_CLIP", ".."]

from data.old_polyvore_text_image_dm import TextImageDataset
from clip.simple_tokenizer import SimpleTokenizer
from clip.clip import tokenize
from tools import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# %%
data_dir = "../data"
image_dir = osp.join(data_dir, "images")
metadata_file = osp.join(data_dir, "polyvore_item_metadata.json")

# %%
metadatas = load_json(metadata_file)
item_ids = list(metadatas.keys())

# %%
n_sample = 5

# %%
img_desc_pairs = []
count = 0

while count < n_sample:
    item_id = random.sample(item_ids, 1)[0]
    metadata = metadatas[item_id]
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    if len(description) == 0:
        continue
    image_name = item_id + ".jpg"
    image_path = osp.join(image_dir, image_name)
    image = load_image(image_path)
    desc = f"Title: {title}\nDescription: {description}"
    img_desc_pairs.append((image, desc))
    count += 1

display_image_with_desc_grid(img_desc_pairs, n_sample, n_rows=1)

# %%
