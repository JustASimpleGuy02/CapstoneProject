# %%
import os
import os.path as osp
from glob import glob
import random
import importlib
import re

import cv2

from reproducible_code.tools import image_io, io, plot
# from apis import search_fclip
# from fashion_clip.fashion_clip import FashionCLIP

# importlib.reload(image_io)
# importlib.reload(search_fclip)

# %%
data_dir = "/home/dungmaster/Datasets/Deep-Fashion"
# img_dir = osp.join(data_dir, "img")
# img_paths = glob(osp.join(img_dir, "*/*.jpg"))
# len(img_paths)

# %%
list_bbox = io.load_txt(
    osp.join(data_dir, "Anno_coarse", "list_bbox.txt")
)[2:]
list_bbox

# %%
list_img = [osp.join(data_dir, bbox.split()[0]) for bbox in list_bbox]
list_bboxes = [bbox.split()[1:] for bbox in list_bbox]
del list_bbox

# %%
print(len(list_img))
print(len(list_bboxes))

# %%
list_img

# %%
list_bboxes = list(map(
    lambda x: [int(_) for _ in x], list_bboxes 
))    
list_bboxes

# %%
num_sample = 16
rand_inds = random.sample(range(0, len(list_img)-1), num_sample)
rand_inds

# %%
imgs = []

for ind in rand_inds:
    img = image_io.load_image(list_img[ind])
    bbox = list_bboxes[ind]
    tl, br = bbox[:2], bbox[2:]
    img = cv2.rectangle(img, tl, br, (255, 0, 0), 2)
    imgs.append(img)

plot.display_multiple_images(imgs, 4, 24, num_sample, 512)

# %%
