# %%
import os
import os.path as osp
from glob import glob
import random
import importlib
import re

import cv2

from reproducible_code.tools import image_io, io, plot

importlib.reload(image_io)

# %%
data_dir = "/home/dungmaster/Datasets/Deep-Fashion"
list_imgs = io.load_txt(
    osp.join(data_dir, "Anno_fine", "train.txt")
)
print(len(list_imgs))
list_imgs
    
# %%
list_bbox = io.load_txt(
    osp.join(data_dir, "Anno_fine", "train_bbox.txt")
)
print(len(list_bbox))
list_bbox

# %%
list_img = [osp.join(data_dir, ) for bbox in list_imgs]
list_bboxes = [bbox.split()for bbox in list_bbox]

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

plot.display_multiple_images(imgs, 4, 24, 512)

# %%
