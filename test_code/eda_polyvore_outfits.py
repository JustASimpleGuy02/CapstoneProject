# %%
import os
import os.path as osp
import sys

sys.path.append("../")
import random
from pprint import pprint
from tools import *

# %%
data_dir = "../data/polyvore"
image_dir = osp.join(data_dir, "images")

# %%
outfit_titles = load_json(osp.join(data_dir, "polyvore_outfit_titles.json"))

# %%
train_disjoin_outfit_items = load_json(
    osp.join(data_dir, "disjoint", "train.json")
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
idx = random.randint(0, len(train_disjoin_outfit_items) - 1)
set_id = train_disjoin_outfit_items[idx]["set_id"]
set_items = train_disjoin_outfit_items[idx]["items"]
set_title = outfit_titles[set_id]

print(set_title)
pprint(set_items)

# %%
outfit_data = train_disjoin_outfit_items
n_outfits = len(outfit_data)
n_sample = 10
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
