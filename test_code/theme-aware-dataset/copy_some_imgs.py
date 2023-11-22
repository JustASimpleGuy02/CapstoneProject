# %%
import os
import os.path as osp
from glob import glob
import random
from shutil import copy

from reproducible_code.tools import image_io, io

# %% 
data_dir = "/home/dungmaster/Datasets/Fashion-Outfits-Theme-Aware"
outfits_dir = osp.join(data_dir, "validation")
img_paths = glob(osp.join(outfits_dir, "*.jpg"))
len(img_paths)

# %%
item_paths = [p for p in img_paths if "outfit_" not in p]
len(item_paths)

# %%
img_dir = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/CapstoneProject/data/theme-aware-dataset/test_images"
io.create_dir(img_dir)

# %%
num_sample = 300
num_splits = 10
num_img_each_split = num_sample / num_splits

random.shuffle(img_paths)
count = 0

new_img_dir = osp.join(img_dir, '1')
io.create_dir(new_img_dir)

for path in item_paths:
    try:
        img = image_io.load_image(path, backend="cv2")
    except Exception:
        continue

    if count % num_img_each_split == 0 and count > 0:
        dir_name = str(int(count/num_img_each_split+1))
        new_img_dir = osp.join(img_dir, dir_name)
        io.create_dir(new_img_dir)
        
    name = osp.basename(path)
    new_path = osp.join(new_img_dir, name)
    copy(path, new_path)

    count += 1
    
    if count == num_sample:
        break

# %%
