# %%
import os
import os.path as osp
import sys

sys.path += ["../train_CLIP", ".."]
import random
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import *

import h5py

# %%
h5_file = "../data/fashion-gen/fashiongen_256_256_validation.h5"
f = h5py.File(h5_file, "r")
keys = list(f.keys())

# %%
for k in keys:
    print(f"{k}: {f[k][0]}")

# %%
idx = 21
test_image = f["input_image"][idx]
test_description = f["input_description"][idx][0].decode("UTF-8")
print(test_description)
print(type(test_description))
plt.imshow(test_image)

# %%
n_images = len(f["input_image"])
n_sample = 10
# rand_inds = random.sample(range(0, n_images-1), n_sample)
rand_inds = range(10)
img_desc_pairs = []

for idx in rand_inds:
    img = f["input_image"][idx]
    desc = f["input_description"][idx][0]
    desc = str(desc)
    img_desc_pairs.append((img, desc))

display_image_with_desc_grid(
    img_desc_pairs, n_sample, n_rows=2, figsize=(20, 10)
)

# %%
