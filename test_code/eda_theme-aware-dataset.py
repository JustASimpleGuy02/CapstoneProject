# %%
import os
import os.path as osp
import sys

import random
from pprint import pprint
import importlib

# from functools import map

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")

from tools import image, load_image, load_json, display_image_sets

importlib.reload(image)

sns.set_theme()

# %% Loading data
data_dir = "/home/dungmaster/Datasets/Fashion_Outfits_Theme_Aware"
outfits_dir = osp.join(data_dir, "outfits")

# %%
outfit_descriptions = load_json(
    osp.join(data_dir, "theme_aware_dataset_descriptions.json"), verbose=True
)
outfit_ids = list(outfit_descriptions.keys())

# %%
n_outfits = len(outfit_ids)
n_sample = 1
rand_inds = random.sample(range(0, n_outfits - 1), n_sample)

sample_images = []
sample_outfit_titles = []

for ind in rand_inds:
    outfit_id = outfit_ids[ind]
    item_images = []
    outfit_images = []

    outfit_dir = osp.join(outfits_dir, outfit_id)
    outfit_info = outfit_descriptions[outfit_id]
    outfit_meta = load_json(osp.join(outfit_dir, outfit_id + ".json"))

    outfit_title = outfit_info["en_Outfit_Name"]
    outfit_desc = outfit_info["en_Outfit_Description"]
    outfit_style = outfit_info["en_Outfit_Style"]
    outfit_occasion = outfit_info["en_Outfit_Occasion"]

    outfit_text = f"Description: {outfit_desc} Name: {outfit_title}. Style: {outfit_style}. Occasion: {outfit_occasion}."

    for item_info in outfit_meta["Items"]:
        image_path = osp.join(outfit_dir, item_info["Image"])
        image = np.array(load_image(image_path))
        if len(image.shape) < 3:
            image = image[..., np.newaxis].repeat(3, -1)
        item_images.append(image)
    item_images = np.hstack(item_images)

    for outfit_image in outfit_meta["Outfit_Images"]:
        outfit_image_path = osp.join(outfit_dir, outfit_image)
        outfit_image = np.array(load_image(outfit_image_path))
        outfit_images.append(outfit_image)
    outfit_images = np.hstack(outfit_images)

    # outfit_images.resize(item_images.shape)
    # print(item_images.shape[::-1])
    # print(outfit_images.shape)

    sample_images += [item_images, outfit_images]
    # sample_outfit_titles += [outfit_title, ""]

display_image_sets(sample_images, title=outfit_text)

# %%
outfit_id

# %%
