# %%
import os.path as osp
from glob import glob
import sys

import random
from pprint import pprint
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tqdm.pandas()
sys.path.append("../")
from reproducible_code.tools import load_json, save_json, load_csv, to_csv, plot_attribute_frequency

# %%
name = lambda x: Path(x).name

data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfit_dirs = glob(osp.join(data_dir, "outfits", "*/"))
len(outfit_dirs)

# %%
cates_cn2en = load_json("../data/theme_aware_fashion_category_cn2en.json")
cates = list(cates_cn2en.values())
cate2imgs = {cate: None for cate in cates}

# %%
# count = 0
# df_outfit_items = pd.DataFrame(columns=[["outfit_id"] + cates])
# invalid_cate_img = []

# for outfit_path in tqdm(outfit_dirs):
#     outfit_id = name(outfit_path)
#     metadata = load_json(osp.join(outfit_path, outfit_id + ".json"))
#     item_infos = metadata["Items"]
#     cate_imgs = cate2imgs.copy()

#     for item_info in item_infos:
#         cate_tag = ""

#         image_name = item_info["Image"]
#         cate_tagl = item_info["Tags"]["97"]

#         if len(cate_tagl) == 0:
#             invalid_cate_img.append(image_name)
#             continue
#         else:
#             cate_tag = cates_cn2en[cate_tagl[0]["label_name"]]

#         cate_imgs[cate_tag] = image_name

#     df_outfit_items.loc[len(df_outfit_items.index)] = [outfit_id] + list(cate_imgs.values())

#     # count += 1
#     # if count >= 5:
#     #     break

# # %%
# df_outfit_items.fillna(-1, inplace=True)
# df_outfit_items.head(5)

# # %%
# to_csv("../data/theme_aware_outfit_items.csv", df_outfit_items)

# %%
df_items = pd.DataFrame(columns=["image", "sub_category"])

for outfit_path in tqdm(outfit_dirs):
    outfit_id = name(outfit_path)
    metadata = load_json(osp.join(outfit_path, outfit_id + ".json"))
    item_infos = metadata["Items"]
    cate_imgs = cate2imgs.copy()

    for item_info in item_infos:
        cate_tag = ""

        image_name = item_info["Image"]
        cate_tagl = item_info["Tags"]["97"]

        if len(cate_tagl) == 0:
            invalid_cate_img.append(image_name)
            continue
        else:
            cate_tag = cates_cn2en[cate_tagl[0]["label_name"]]

        df_items.loc[len(df_items.index)] = [image_name, cate_tag]

# %%
subcates = list(set(df_items["sub_category"].tolist()))
subcates

# %%
subcates = df_items["sub_category"].value_counts().reset_index()
