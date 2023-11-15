# %%
import os
import os.path as osp
from glob import glob

# %%
data_dir = "/home/dungmaster/Datasets/Deep-Fashion"
img_dir = osp.join(data_dir, "img")
images = glob(osp.join(img_dir, "*/*.jpg"))
len(images)

# %%
