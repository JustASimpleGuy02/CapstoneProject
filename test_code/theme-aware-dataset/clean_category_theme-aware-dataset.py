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
from reproducible_code.tools import io, plot
from translatepy.translators.google import GoogleTranslateV2

import importlib
importlib.reload(plot)

# %%
name = lambda x: Path(x).name
get_outfit = lambda x: x.split('_')[0]
gtranslate = GoogleTranslateV2()

# %%
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfit_dirs = glob(osp.join(data_dir, "outfits", "*/"))
len(outfit_dirs)

# %%
cates_cn2en = io.load_json("../../data/theme_aware_fashion_category_cn2en.json")
print(len(cates_cn2en))
cates_cn2en

# %%
subcates_cn2en = io.load_json("../../data/theme_aware_fashion_subcategory_cn2en.json")
print(len(subcates_cn2en))
subcates_cn2en

# # %%
# cates = list(cates_cn2en.values())
# cate2imgs = {cate: None for cate in cates}

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
# invalid_cate_img

# # %%
# to_csv("../data/theme_aware_outfit_items.csv", df_outfit_items)

# # %%
# df_items = pd.DataFrame(columns=["image", "category", "subcategory"])

# for idx, outfit_path in tqdm(enumerate(outfit_dirs)):
#     outfit_id = name(outfit_path)
#     metadata = load_json(osp.join(outfit_path, outfit_id + ".json"))
#     item_infos = metadata["Items"]

#     for item_info in item_infos:
#         image_name = item_info["Image"]
#         tag_info = item_info["Tags"]

#         tag_97 = tag_info.get("97", [])
#         cate = tag_97[0]["label_name"] if len(tag_97) > 0 else ''

#         tag_98 = tag_info.get("98", [])
#         tag_99 = tag_info.get("99", [])
#         subcate = tag_99[0]["label_name"] if len(tag_99) > 0 else tag_98[0]["label_name"] if len(tag_98) > 0 else ''
        
#         df_items.loc[len(df_items.index)] = [image_name, cate, subcate]

#     # if idx == 5:
#     #     break

# print(len(df_items))
# df_items.head()

# # %%
# subcates = list(set(df_items["subcategory"].tolist()))
# io.save_txt(subcates, "../../data/theme-aware_subcates.txt")

# # %%
# subcates_cn2en = {cn: gtranslate.translate(cn, "English").result for cn in subcates}
# subcates_cn2en

# # %%
# io.save_json(subcates_cn2en, "../../data/theme_aware_fashion_subcategory_cn2en.json")

# # %%
# df_items["en_subcategory"] = df_items["subcategory"].apply(
#     lambda x: subcates_cn2en[x]
# )

# # %%
# mask = df_items["category"].notna()
# df_items.loc[mask, "en_category"] = df_items.loc[mask, "category"].apply(
#     lambda x: cates_cn2en[x]
# )
# df_items.head()

# # %%
# df_items["en_subcategory"] = df_items["sub_category"].apply(
#     lambda x: subcates_cn2en[x].lower() if len(x) != 0 else ''
# )
# df_items["image_path"] = df_items["image"].apply(
#     lambda x: osp.join(data_dir, "outfits", get_outfit(x), x)
# )
# print(df_items.loc[0].image_path)
# df_items.head(5)

# %%
path = "../../data/theme-aware_item_categories.csv"
# io.to_csv(path, df_items)
df_items = io.load_csv(path)
print(len(df_items))
print("Number of items with no category:", sum(df_items["en_category"].isna()))
print("Number of items with no subcategory:", sum(df_items["en_subcategory"].isna()))
df_items.head()

# %%
subcates = list(set(df_items["en_subcategory"].tolist()))
print(len(subcates))
subcates

# %% Check outerwear categories
outerwears = ["cardigan", "sweater", "sweatshirt",
              "jacket", "coat", "suit", "vest"]
mask = df_items["en_subcategory"].notna()
subcate_valid = df_items[mask]

new_col = "outerwear"
df_items[new_col] = False

df_items.loc[mask, new_col] = subcate_valid["en_subcategory"].apply(
    lambda x: sum([c in x for c in outerwears]) != 0
)
df_items.head(5)

# %%
outerwear_items = df_items[df_items[new_col]][["image_path", "en_subcategory"]]
print(len(outerwear_items))
outerwear_items.head()

# %%
sample_outerwears = outerwear_items.sample(16)
img_subcate_dict = dict(zip(sample_outerwears.image_path, sample_outerwears.en_subcategory))

img_paths = list(img_subcate_dict.keys())

plot.display_multiple_images(img_paths, 4, 12, 16, 512, img_subcate_dict)

# %% Check skirt categories
cate = "skirt suit"
chosen_cate_items = df_items[df_items["en_category"] == cate]
chosen_cate_items.head()

# %%
cate = "en_subcategory"
# subcates_freq = chosen_cate_items[cate].value_counts().reset_index()
# sns.barplot(subcates_freq, x=cate, y="index")
plot.plot_attribute_frequency(chosen_cate_items, cate)

# %%
