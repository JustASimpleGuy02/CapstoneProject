# %%
import os
import os.path as osp
import sys

import random
from pprint import pprint
import importlib
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

sys.path += ["../"]
from reproducible_code.tools import io, plot, image_io
importlib.reload(plot)

sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
tqdm.pandas()

# %% Loading data
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "outfits")

# %% Load csv
df_outfit_meta = pd.read_csv(
    osp.join(data_dir, "outfit_meta.csv"), index_col=None
)
df_outfit_meta.head(5)

# %%
df_outfit_meta.isna().sum()

# %%
df = df_outfit_meta["en_Outfit_Description"].dropna().apply(lambda x: len(x)).rename("description length")
sns.histplot(df)

# %%
df = df_outfit_meta["en_Outfit_Name"].dropna().apply(lambda x: len(x)).rename("name length")
sns.histplot(df)

# %%
field = "en_Outfit_Style"
df = df_outfit_meta[field].dropna().value_counts().reset_index()
ax = sns.barplot(df, x=field, y="index")
ax.bar_label(ax.containers[0], fontsize=10);

# %%
field = "en_Outfit_Occasion"
df = df_outfit_meta[field].dropna().value_counts().reset_index()
ax = sns.barplot(df, x=field, y="index")
ax.bar_label(ax.containers[0], fontsize=10);

# %%
field = "en_Outfit_Fit"
df = df_outfit_meta[field].dropna().value_counts().reset_index()
ax = sns.barplot(df, x=field, y="index")
ax.bar_label(ax.containers[0], fontsize=10);

# %%
field = "en_Outfit_Gender"
df = df_outfit_meta[field].dropna().value_counts().reset_index()
ax = sns.barplot(df, x=field, y="index")
ax.bar_label(ax.containers[0], fontsize=10);

# %%
outfit_descriptions = io.load_json(
    osp.join(data_dir, "theme_aware_dataset_descriptions.json"), verbose=True
)
outfit_ids = list(outfit_descriptions.keys())

# %% Display outfits with their descriptions
n_outfits = len(outfit_ids)
n_sample = 8
# sample_outfit_ids = random.sample(outfit_ids, n_sample)
sample_outfit_ids = ["5858"]
new_sizes = (224, 224)

sample_images = []
sample_outfit_titles = []
grey_images = []

for outfit_id in sample_outfit_ids:
    item_images = []
    outfit_images = []

    outfit_dir = osp.join(outfits_dir, outfit_id)
    outfit_info = outfit_descriptions[outfit_id]
    outfit_meta = io.load_json(osp.join(outfit_dir, outfit_id + ".json"))

    outfit_title = outfit_info["en_Outfit_Name"]
    outfit_desc = outfit_info["en_Outfit_Description"]
    outfit_style = outfit_info["en_Outfit_Style"]
    outfit_occasion = outfit_info["en_Outfit_Occasion"]

    outfit_text = f"Description: {outfit_desc} Name: {outfit_title}. Style: {outfit_style}. Occasion: {outfit_occasion}."

    for item_info in outfit_meta["Items"]:
        image_path = osp.join(outfit_dir, item_info["Image"])

        try:
            # image = np.array(Image.open(image_path))
            image = image_io.load_image(
                image_path,
                # toRGB=False
            )
        except Exception:
            continue

        if image is None:
            continue
        
        sizes = image.shape
    
        if len(sizes) == 3:
            if sizes[-1] == 4:
                print("4-channel image")
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            print(f"grey image")
            image = image[..., np.newaxis].repeat(3, -1)
            grey_images.append(image)
        
        image = cv2.resize(image, new_sizes)
        item_images.append(image)

    item_images = np.vstack(item_images)
    ih, iw, _ = item_images.shape

    for outfit_image in outfit_meta["Outfit_Images"]:
        outfit_image_path = osp.join(outfit_dir, outfit_image)
        outfit_image = image_io.load_image(
            outfit_image_path,
            # toRGB=False
        )
        # outfit_image = np.array(Image.open(outfit_image_path))
        outfit_images.append(outfit_image)

    outfit_images = np.vstack(outfit_images)
    oh, ow, _ = outfit_images.shape

    outfit_images = cv2.resize(outfit_images, (int(ih * ow/oh), ih))
    combined_image = np.hstack((item_images, outfit_images))

    sample_images.append(combined_image)
    sample_outfit_titles.append(outfit_text)

# display_image_sets(sample_images, title=outfit_text)
plot.display_multiple_images(
    sample_images,
    grid_nrows=1,
    fig_size=12,
    titles=sample_outfit_titles,
    fontsize=10,
    axes_pad=2.,
    line_length=8
)

# %%
img_name = "12597/12597_12003_37184955938.jpg"
img_name = "8935/8935_1356_30236405816.jpg"
img_path = osp.join(outfits_dir, img_name)
print(img_path)
img = image_io.load_image(img_path)
img = np.array(Image.open(img_path))
img.shape

# %%
