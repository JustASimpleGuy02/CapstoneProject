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
from reproducible_code.tools import plot, io

importlib.reload(plot)

sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})
tqdm.pandas()

# %% Loading data
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "outfits")

# # %%
# df = pd.read_csv("../../data/theme_aware_outfit_items.csv")
# df.head(5)

# # %% Clean "flat" category
# cate = "flat"
# cate_valid_df = df[df[cate] != "-1"]
# img_names = cate_valid_df[cate].tolist()
# img_paths = [osp.join(data_dir, "outfits", img.split('_')[0], img) for img in img_names]

# # %%
# plot.plot_multiple_images(img_paths, grid_nrows=2, fig_size=12)

# # %%
# df["shoe"] = np.where(
#     df[cate].eq("-1"), df["footwear"], df[cate]
# )

# # %%
# df.drop([cate, "footwear"], axis=1, inplace=True)
# df.head(5)

# # %%
# df.rename(columns={"shoe": "footwear", "Tops": "top"}, inplace=True)
# df.head(5)

# # %%
# df.loc[[1385, 2300]]

# # %% Clean "Package class" category
# cate = "Package class"
# cate_valid_df = df[df[cate] != "-1"]
# img_names = cate_valid_df[cate].tolist()
# img_paths = [osp.join(data_dir, "outfits", img.split('_')[0], img) for img in img_names]
# len(img_paths)

# # %%
# plot.plot_multiple_images(
#     img_paths,
#     grid_nrows=4,
#     fig_size=12,
#     num_images_to_plot=16
# )

# # %%
# df.rename(columns={cate: "bag"}, inplace=True)
# df.head(5)

# # %% Clean "Accessories (exclusive)" category
# cate = "Accessories (exclusive)"
# cate_valid_df = df[df[cate] != "-1"]
# img_names = cate_valid_df[cate].tolist()
# img_paths = [osp.join(data_dir, "outfits", img.split('_')[0], img) for img in img_names]
# len(img_paths)

# # %%
# plot.plot_multiple_images(
#     img_paths,
#     grid_nrows=4,
#     fig_size=12,
#     num_images_to_plot=16
# )

# # %%
# df.rename(columns={cate: "accessory"}, inplace=True)
# df.head(5)

# %%
out_path = "../../data/clean_theme_outfit_items_v1.csv" 
io.to_csv(out_path, df)

# %%
df = pd.read_csv(out_path)
print(len(df))
df.head(5)

# %% Check bottom items
bottom_df = df[["skirt suit", "trousers"]]
bottom_df.head()
# %%
bottom_df["exclusive"] = np.logical_or(
    (bottom_df["skirt suit"] == "-1"),
    (bottom_df["trousers"] == "-1")
)
bottom_df.head(20)

# %%
invalid_index = bottom_df[~bottom_df["exclusive"]].index.tolist()
invalid_index

# %%
cate = "skirt suit"
cate_valid_df = df[df[cate] != "-1"]
img_names = cate_valid_df[cate].tolist()
img_paths = [osp.join(data_dir, "outfits", img.split('_')[0], img) for img in img_names]
len(img_paths)

# %%
cates = df.columns.to_list()[1:]
cates

# %%
plot.plot_multiple_images(
    img_paths,
    grid_nrows=4,
    fig_size=12,
    num_images_to_plot=16
)

# %%
invalid_bottom_df = df.loc[invalid_index].reset_index()
invalid_bottom_df

# %%
outfit_descriptions = io.load_json(
    osp.join(data_dir, "theme_aware_dataset_descriptions.json"), verbose=True
)

# %%
# rand_ind = random.sample(range(len(invalid_bottom_df)), 1)[0]
rand_ind = 4
print(rand_ind)
row = invalid_bottom_df.loc[rand_ind]

sample_images = []
sample_titles = []
image_2ds = []

titles = []
outfit_id = str(row.outfit_id)
print(outfit_id)
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

for c in cates:
    item = row[c]

    if item == "-1":
        continue

    titles.append(c)
    image_path = osp.join(outfit_dir, item)
    image = np.array(Image.open(image_path))
    sizes = image.shape

    if len(sizes) == 3:
        if sizes[-1] == 4:
            print("4-channel image")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        print(f"gray image in {c} category")
        image = image[..., np.newaxis].repeat(3, -1)

    item_images.append(image)

item_images = np.hstack(item_images)

for outfit_image in outfit_meta["Outfit_Images"]:
    outfit_image_path = osp.join(outfit_dir, outfit_image)
    outfit_image = np.array(Image.open(outfit_image_path))
    outfit_images.append(outfit_image)

outfit_images = np.hstack(outfit_images)

sample_images += [item_images, outfit_images]
sample_titles.append(titles)

print(titles)
plot.display_image_sets(sample_images, title=outfit_text, descriptions=sample_titles)

# %%
if len(image_2ds) > 0:
    plt.imshow(np.hstack(image_2ds), cmap="gray")

# %% Remove outfit with the id 7354 because its images are unrelated to outfit
df = df[df["outfit_id"] != 7354]
len(df)

# %%
df["top"].tolist()

# %%
