# %%
import os
import os.path as osp
from glob import glob
import random
import importlib
import re
from tqdm import tqdm

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from reproducible_code.tools import image_io, io, plot

# importlib.reload(image_io)

# %%
data_dir = "/home/dungmaster/Datasets/Deep-Fashion"

# # %%
# img2attr_json = "../../data/deep-fashion-dataset/img2attr.json"
# # io.save_json(img2attr, img2attr_json)
# img2attr = io.load_json(img2attr_json)
# print(len(img2attr))

# # # %% [markdown]
# # ## Category 
# list_cate = io.load_txt(
#     osp.join(data_dir, "Anno_coarse", "list_category_cloth.txt")    
# )[2:]
# list_cates = [pair.split()[0] for pair in list_cate]
# list_cates

# # %%
# img2cate = io.load_txt(
#     osp.join(data_dir, "Anno_coarse", "list_category_img.txt")    
# )[2:]

# for pair in tqdm(img2cate):
#     pair = pair.split()
#     img = pair[0]
#     cate = list_cates[int(pair[1])]
#     attr = img2attr[img]
#     img2attr[img] = {"category": cate, "attributes": attr}

# # %% [markdown]
# ## Category 
list_cate = io.load_txt(
    osp.join(data_dir, "Anno_coarse", "list_category_cloth.txt")    
)[2:]
list_cates = [pair.split()[0] for pair in list_cate]
list_cates

# %%
img2cate = io.load_txt(
    osp.join(data_dir, "Anno_coarse", "list_category_img.txt")    
)[2:]
all_cates = []

for pair in tqdm(img2cate):
    pair = pair.split()
    img = pair[0]
    cate = list_cates[int(pair[1])]
    all_cates.append(cate)

len(all_cates)
all_cates

# %%
freqs = plot.plot_attribute_frequency(all_cates, "category", 10, 10, idx_ranges=[0, 10])

# %%
len(freqs)

# # %% [markdown]
# # ## Attributes
# f = open(osp.join(data_dir, "Anno_coarse", "list_attr_cloth.txt"), 'r')
# attr_names = [' '.join(line.split()[:-1]) for line in f.readlines()[2:]]
# attr_names

# # %%
# f = open(osp.join(data_dir, "Anno_coarse", "list_attr_img.txt"), 'r')
# img2attr = {}

# for line in tqdm(f.readlines()[2:]):
#     temp = line.split()
#     img = temp[0]
#     attr_mask = temp[1:]
#     attrs = [attr_names[i] for i in range(len(attr_mask)) if attr_mask[i] == '1']

#     img2attr[img] = attrs

#     # if len(attrs) > 0:
#     #     print(list_img)
#     #     print(attrs)
#     #     break

# %% [markdown]
# ## Attributes
f = open(osp.join(data_dir, "Anno_coarse", "list_attr_cloth.txt"), 'r')
attr_names = [' '.join(line.split()[:-1]) for line in f.readlines()[2:]]
attr_names

# %%
f = open(osp.join(data_dir, "Anno_coarse", "list_attr_img.txt"), 'r')
attr_freqs = {}

for line in tqdm(f.readlines()[2:]):
    temp = line.split()
    img = temp[0]
    attr_mask = temp[1:]
    attrs = [i for i in range(len(attr_mask)) if attr_mask[i] == '1']
    for attr in attrs:
        attr_freqs[attr] = attr_freqs.get(attr, 0) + 1

len(attr_freqs)

# %%
attr_counts = sorted(list(attr_freqs.values()))

# %%
plt.plot(attr_counts)
plt.xlabel("attributes")
plt.ylabel("count")

# %%
img2meta_json = "../../data/deep-fashion-dataset/img2meta.json"
# io.save_json(img2attr, img2meta_json)
img2meta = io.load_json(img2meta_json, verbose=True)
img2meta

# %% [markdown]
# ## Bboxes
list_bbox = io.load_txt(
    osp.join(data_dir, "Anno_coarse", "list_bbox.txt")
)[2:]

# %%
# list_img = [osp.join(data_dir, bbox.split()[0]) for bbox in list_bbox]
list_img = [bbox.split()[0] for bbox in list_bbox]
list_bboxes = [bbox.split()[1:] for bbox in list_bbox]
del list_bbox

# %%
print(len(list_img))
print(len(list_bboxes))

# %%
list_bboxes = list(map(
    lambda x: [int(_) for _ in x], list_bboxes 
))    

# %%
num_sample = 16
rand_inds = random.sample(range(0, len(list_img)-1), num_sample)
imgs = []
titles = []

for ind in rand_inds:
    img_path = list_img[ind]
    print(img_path)
    meta = img2meta[img_path]

    cate = meta["category"]
    attr = ','.join(meta["attributes"])

    full_img_path = osp.join(data_dir, img_path)
    img = image_io.load_image(full_img_path)
    bbox = list_bboxes[ind]
    tl, br = bbox[:2], bbox[2:]
    img = cv2.rectangle(img, tl, br, (255, 0, 0), 2)

    category = "Category: " + cate 
    attr = "Attributes: " + attr
    title = category + "\n" + attr

    imgs.append(img)
    titles.append(title)

plot.display_multiple_images(
    imgs,
    4,
    24,
    512,
    titles=titles,
    fontsize=12,
    axes_pad=0.7,
    line_length=6,
    sep=','
)

# %%
