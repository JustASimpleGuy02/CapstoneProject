# %%
import os
import os.path as osp
import sys

import random
from pprint import pprint

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("../")
from tools import *

# %%
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "images")

# %%
outfit_titles = load_json(osp.join(data_dir, "polyvore_outfit_titles.json"), verbose=True)

# %%
item_metadata = load_json(osp.join(data_dir, "polyvore_item_metadata.json"), verbose=True)

# %%
train_disjoin_outfit_items = load_json(
    osp.join(data_dir, "disjoint", "train.json"), verbose=True
)
valid_disjoin_outfit_items = load_json(
    osp.join(data_dir, "disjoint", "valid.json")
)
test_disjoin_outfit_items = load_json(
    osp.join(data_dir, "disjoint", "test.json")
)
train_nondisjoin_outfit_items = load_json(
    osp.join(data_dir, "nondisjoint", "train.json")
)
val_nondisjoin_outfit_items = load_json(
    osp.join(data_dir, "nondisjoint", "valid.json")
)
test_nondisjoin_outfit_items = load_json(
    osp.join(data_dir, "nondisjoint", "test.json")
)

(
    len(train_disjoin_outfit_items)
    + len(valid_disjoin_outfit_items)
    + len(test_disjoin_outfit_items)
    + len(train_nondisjoin_outfit_items)
    + len(val_nondisjoin_outfit_items)
    + len(test_nondisjoin_outfit_items)
)

# %%
doubt_cates = ["bottoms", "shoes"]
id2cates = pd.read_csv(osp.join(data_dir, "categories.csv"),
                       index_col=None,
                       names=["id", "category", "semantic_category"])
id2cates.head()
shoeid = list(set(id2cates[id2cates["semantic_category"] == "shoes"]["id"]))
bottomid = list(set(id2cates[id2cates["semantic_category"] == "bottoms"]["id"]))
shoeid

# %%
shoe_freqs = {}
bottom_freqs = {}
chosen_ids = shoeid + bottomid
for meta in tqdm(item_metadata.values()):
    cate_id = int(meta["category_id"])
    if cate_id not in chosen_ids:
        continue
    cate = id2cates[id2cates["id"] == cate_id]["category"].iloc[0]
    if cate_id in shoeid:
        shoe_freqs[cate] = shoe_freqs.get(cate, 0) + 1
    else:
        bottom_freqs[cate] = bottom_freqs.get(cate, 0) + 1

shoe_freqs, bottom_freqs

# %%
shoe_kv = {"category": list(shoe_freqs.keys()),
           "frequency": list(shoe_freqs.values())
           }
shoe_df = pd.DataFrame(shoe_kv)
shoe_df = shoe_df.sort_values("frequency", ascending=False).head(5)

my_plot = sns.barplot(data=shoe_df, x="category", y="frequency")
plt.title("Top shoe categories frequency")
my_plot.set_xticklabels(my_plot.get_xticklabels())
plt.savefig("../save_figs/shoe_categories_frequency.png")

# %%
bottom_kv = {"category": list(bottom_freqs.keys()),
             "frequency": list(bottom_freqs.values())
             }
bottom_kv

bottom_df = pd.DataFrame(bottom_kv)
bottom_df = bottom_df.sort_values("frequency", ascending=False).head(5)

my_plot = sns.barplot(data=bottom_df, x="category", y="frequency")
plt.title("Top bottom categories frequency")
my_plot.set_xticklabels(my_plot.get_xticklabels())
plt.savefig("../save_figs/bottom_categories_frequency.png")

# %%
idx = random.randint(0, len(train_disjoin_outfit_items) - 1)
set_id = train_disjoin_outfit_items[idx]["set_id"]
set_items = train_disjoin_outfit_items[idx]["items"]
set_title = outfit_titles[set_id]

print(set_title)
pprint(set_items)

# %%
outfit_data = train_disjoin_outfit_items
n_outfits = len(outfit_data)
n_sample = 5
rand_inds = random.sample(range(0, n_outfits - 1), n_sample)

sample_images = []
sample_set_titles = []

for ind in rand_inds:
    images = []

    set_id = outfit_data[ind]["set_id"]
    set_items = outfit_data[ind]["items"]
    set_title = outfit_titles[set_id]
    set_title = set_title["url_name"] + ". " + set_title["title"]

    for item in set_items:
        image_path = osp.join(image_dir, item["item_id"] + ".jpg")
        image = load_image(image_path, backend="pillow")
        images.append(image)

    sample_images.append(images)
    sample_set_titles.append(set_title)

display_image_sets(sample_images, sample_set_titles, fontsize=7)

# %%
