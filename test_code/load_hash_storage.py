# %%
import os
import os.path as osp
import pickle
from shutil import copy
from tqdm import tqdm

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning"
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "images")
new_image_dir = osp.join(
    project_dir,
    "HangerAI_outfits_recommendation_system",
    "CapstoneProject",
    "static"
)

hashes_file = osp.abspath(
    osp.join(
        project_dir,
        "HangerAI_outfits_recommendation_system",
        "storages",
        "hanger_apparels_100.pkl",
    )
)

# %%
pkl_file = open(hashes_file, "rb")
hashes_polyvore = pickle.load(pkl_file)[1]

# %%
image_paths = list(hashes_polyvore.keys())

for path in tqdm(image_paths):
    src_img_path = osp.join(image_dir, path)
    dst_img_path = osp.join(new_image_dir, path)
    try:
        copy(src_img_path, dst_img_path)
    except Exception as e:
        continue

# %%
sample = hashes_polyvore[image_paths[21]]
sample.keys()

# %%
