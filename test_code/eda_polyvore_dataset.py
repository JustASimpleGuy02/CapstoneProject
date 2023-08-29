# %%
import os
import sys
import os.path as osp
sys.path.append(osp.dirname(os.getcwd())) # append parent directory
import random
from tools import *

import cv2
import pandas as pd
import polars as pl

# %%
data_dir = "../data"
image_dir = osp.join(data_dir, "images")
image_desc_file = osp.join(data_dir, "polyvore_img_desc.csv")

# %%
df = pl.read_csv(image_desc_file)
text, image = df[12]
text, image = text.item(), image.item()

# %%
text, image

# %%
n_sample = 20
random_indices = [random.randint(0, len(df)-1) for _ in range(n_sample)]
random_indices

# %%
img_desc_pairs = []

for idx in random_indices:
    text, image = df[idx]
    image_path = osp.join(image_dir, image.item())
    image = load_image(image_path)
    img_desc_pairs.append((image, text.item()))
    
# %%
display_image_with_desc_grid(img_desc_pairs, n_sample, n_rows=4)

# %%
len(df)
