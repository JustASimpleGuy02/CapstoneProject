# %%
import os
import os.path as osp
import sys

sys.path += ["../train_CLIP", ".."]
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt
from PIL import Image

import random
from tools import *

# %%
data_dir = "../data/fashion4events/"
image_dir = osp.join(data_dir, "valid")
label_file = osp.join(data_dir, "valid.txt")

image_size = (224, 224)
orig_class_names = [
    "concert",
    "graduation",
    "meeting",
    "mountain-trip",
    "picnic",
    "sea-holiday",
    "ski-holiday",
    "wedding",
    "conference",
    "exhibition",
    "fashion",
    "protest",
    "sport",
    "theater-dance",
]

# %%
# data_lines = open(label_file).readlines()
df = pd.read_csv(
    label_file, names=["path", "directory", "label"], index_col=None
)
df.head()

# %%
df["path"] = df["path"].progress_apply(
    lambda x: str(Path(image_dir) / Path(x).parent.name / Path(x).name)
)
n_items = len(df)
print("Number of items: ", n_items)
df.head()

# %%
print("Unique directories:")
print(df["directory"].value_counts())
print("Unique labels:")
print(df["label"].value_counts())

# %%
rand_ind = random.randint(0, n_items - 1)
rand_row = df.iloc[rand_ind]
test_path = rand_row["path"]
test_label = rand_row["label"]
test_image = Image.open(test_path)
print("Original size:", test_image.size)
test_image = test_image.resize(image_size, Image.Resampling.LANCZOS)
plt.imshow(test_image)
print("Label:", orig_class_names[test_label])

# %%
n_sample = 20
img_desc_pairs = []
rand_inds = random.sample(range(0, n_items - 1), n_sample)

for ind in rand_inds:
    rand_row = df.iloc[ind]
    path = rand_row["path"]
    label = rand_row["label"]
    image = Image.open(path)
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    label = orig_class_names[label]
    img_desc_pairs.append((image, label))

display_image_with_desc_grid(
    img_desc_pairs, n_sample, n_rows=4, figsize=(20, 20)
)

# %%
label = "sport"
label2id = orig_class_names.index(label)
print(label2id)
subset = df[df["label"] == label2id]

n_sample = 20
img_desc_pairs = []
rand_inds = random.sample(range(0, len(subset) - 1), n_sample)

for ind in rand_inds:
    rand_row = subset.iloc[ind]
    path = rand_row["path"]
    label = rand_row["label"]
    image = Image.open(path)
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    label = orig_class_names[label]
    img_desc_pairs.append((image, label))

display_image_with_desc_grid(
    img_desc_pairs, n_sample, n_rows=4, figsize=(20, 20)
)

# %%
