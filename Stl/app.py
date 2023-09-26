# %%
import sys

import os
import os.path as osp
from glob import glob
import random
import time
import requests
import importlib
import pickle

import streamlit as st
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

sys.path.append("/home/dungmaster/Projects/Machine Learning/CapstoneProject")
from tools import load_json, base64_to_image
import apis.search_item_fashion_clip as search_fclip

importlib.reload(search_fclip)

# Hyperparams
random.seed(42)

# par_dir = osp.abspath(osp.join(os.getcwd(), os.pardir))
project_dir = "/home/dungmaster/Projects/Machine Learning"
par_dir = osp.join(project_dir, "CapstoneProject")
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"

hashes_file = osp.abspath(osp.join(
    project_dir,
    "HangerAI_outfits_recommendation_system",
    "storages",
    "hanger_apparels_100.pkl",
))

pkl_file = open(hashes_file, 'rb')
hashes_polyvore = pickle.load(pkl_file)[1]

embeddings_file = osp.join(par_dir, "model_embeddings", "polyvore_10k_v2.txt")
# embeddings_file = osp.join(par_dir, "model_embeddings", "polyvore_100k.txt")

metadata_file = (
    "/home/dungmaster/Datasets/polyvore_outfits/polyvore_item_metadata.json"
)

total_items = len(np.loadtxt(embeddings_file))
top_k = 10
n_cols = 5

image_embeddings = None
save_embeddings = False

json_dict = {"top": 1, "bottom": 1, "bag": 1, "outerwear": 1, "shoe": 1}

name = lambda x: osp.basename(x).split(".")[0]

# %% Execute app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)


# %% Load image data
@st.cache_data
def load_data(image_dir, metadata_file, total_items: int):
    image_paths = glob(osp.join(image_dir, "*.jpg"))
    metadata = load_json(metadata_file)
    # print(len(image_paths))
    image_paths = random.sample(image_paths, total_items)

    return metadata, image_paths


print("Loading image data...")
metadata, image_paths = load_data(image_dir, metadata_file, total_items)
# print(image_paths[:5])


# %% Load image embedding and model
@st.cache_data
def load_embeddings(embeddings_file):
    image_embeddings = None

    if osp.isfile(embeddings_file):
        print(f"Load image embeddings from {embeddings_file}!")
        image_embeddings = np.loadtxt(embeddings_file)
    else:
        print(
            "Embedding file does not exist. Process the images and save embeddings!"
        )
        embeddings_dir = osp.dirname(embeddings_file)
        os.makedirs(embeddings_dir, exist_ok=True)

    return image_embeddings


image_embeddings = load_embeddings(embeddings_file)


# %% Load model
# TODO: using onnx model
@st.cache_resource
def load_model():
    model = FashionCLIP("fashion-clip", approx=False)
    return model


print("Loading model...")
fclip = load_model()


@st.cache_data
def encode_images(image_paths, batch_size=64):
    embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
    return embeddings


@st.cache_data
def encode_prompt(prompt, batch_size=64):
    embeddings = fclip.encode_text([prompt], batch_size=batch_size)
    return embeddings


# %%
# prompt = st.text_input("Search: ")
prompt = "vacation outfit for men"

# Search items
t1 = time.time()
found_image_paths, image_embeddings = search_fclip.search(
    fclip,
    encode_images,
    encode_prompt,
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
    # with cols[int(i % n_cols)]:
    image_path = found_image_paths[i]
    image_id = name(image_path)
    category = metadata[image_id]["semantic_category"]

    if category != "tops":
        continue
    category = category.replace("s", "")
    json_dict[category] = image_id

    response = requests.post(
        url="http://127.0.0.1:3000/items/1/outfits_recommend/",
        json=json_dict,
    )
    print(response.json())

    st.image(image_path, width=250)

# %%
