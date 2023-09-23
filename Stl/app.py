# %%
import sys

import os
import os.path as osp
from glob import glob
import random
import streamlit as st
import time 

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

sys.path.append("/home/dungmaster/Projects/Machine Learning/CapstoneProject")
from apis.search_item_fashion_clip import search

# Hyperparams
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
embeddings_file = "./model_embeddings/polyvore_10k_09-23_v2.txt"

total_items = 10000
top_k = 10
n_cols = 5

image_embeddings = None
save_embeddings = True

# %% Load image data
random.seed(42)

image_paths = glob(osp.join(image_dir, "*.jpg"))
# print(len(image_paths))
image_paths = random.sample(image_paths, total_items)
print(image_paths[:5])

# %% Load image embedding if any
if osp.isfile(embeddings_file):
    print(f"Load image embeddings from {embeddings_file}!")
    image_embeddings = np.loadtxt(embeddings_file)
else:
    print("Embedding file does not exist. Process the images and save embeddings!")
    embeddings_dir = osp.dirname(embeddings_file)
    os.makedirs(embeddings_dir, exist_ok=True)
    save_embeddings = True

# Load model
#TODO: usine onnx model
fclip = FashionCLIP("fashion-clip")

# %%
st.set_page_config(
    page_title="Fashion Garments Search App",
    layout="wide"
)
st.header("Fashion Garments Search App")
prompt = st.text_input('Search: ')

# st.sidebar.header("App Settings")
# top_number = st.sidebar.slider(
#     "Number of Search Results", min_value=1, max_value=30
# )
# picture_width = st.sidebar.slider(
#     "Picture Width", min_value=100, max_value=500
# )

# Search items
t1 = time.time()
found_image_paths, image_embeddings = search(
    fclip,
    prompt=prompt,
    image_paths=image_paths,
    embeddings_file=embeddings_file,
    image_embeddings=image_embeddings,
    top_k=top_k,
    save_embeddings=save_embeddings,
)
t2 = time.time()
print(f"Time: {t2-t1}s")

# %% Display result in 3 columns
cols = st.columns(n_cols)

for i in range(top_k):
    with cols[int(i % n_cols)]:
        st.image(found_image_paths[i], width=250)

# %%
