# %%
import sys
import os
import os.path as osp
import random
import time
import requests
import pickle
from pprint import pprint
from typing import List

import streamlit as st
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP

sys.path.append("/home/dungmaster/Projects/Machine Learning/CapstoneProject")
from tools import load_json
from apis.search_item_fashion_clip import search, FashionRetrieval

# Hyperparams
random.seed(42)

# par_dir = osp.abspath(osp.join(os.getcwd(), os.pardir))
project_dir = "/home/dungmaster/Projects/Machine Learning"
par_dir = osp.join(project_dir, "CapstoneProject")
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
# TODO: change storage pkl file to more data, preferably full polyvore data
hashes_file = osp.abspath(
    osp.join(
        project_dir,
        "HangerAI_outfits_recommendation_system",
        "storages",
        "hanger_apparels_100.pkl",
    )
)

embeddings_file = osp.join(par_dir, "model_embeddings", "polyvore_502.txt")

metadata_file = (
    "/home/dungmaster/Datasets/polyvore_outfits/polyvore_item_metadata.json"
)

image_embeddings = None

json_request = {
    "top": [],
    "bottom": [],
    "bag": [],
    "outerwear": [],
    "shoe": [],
}
# json_request = {"top": 1, "bottom": 1, "bag": 1, "outerwear": 1, "shoe": 1}
cates = [c for c in list(json_request.keys()) if json_request[c] != 0]

top_k = 20
empty_cate_extras = 5
n_cols = len(cates)

name = lambda x: osp.basename(x).split(".")[0]

# %% Execute app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)


# %%
@st.cache_data
def load_image_files(hfile):
    pkl_file = open(hfile, "rb")
    hashes_polyvore = pickle.load(pkl_file)[1]
    image_names = list(hashes_polyvore.keys())
    image_paths = [osp.join(image_dir, name) for name in image_names]
    return image_paths


image_paths = load_image_files(hashes_file)
# image_paths[:5]


# %% Load image data
@st.cache_data
def load_meta(metadata_file):
    metadata = load_json(metadata_file)

    return metadata


metadata = load_meta(metadata_file)


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
def encode_images(image_paths: List[str], batch_size=64):
    embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
    return embeddings


@st.cache_data
def encode_prompt(prompt: List[str], batch_size=64):
    embeddings = fclip.encode_text(prompt, batch_size=batch_size)
    return embeddings


# %%
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "wedding suit for men"
ret = FashionRetrieval(
    model=fclip, image_paths=image_paths, image_embeddings=image_embeddings
)

# Search items
t1 = time.time()
# found_image_paths = search(
#     fclip,
#     encode_images,
#     encode_prompt,
#     prompt=prompt,
#     image_paths=image_paths,
#     embeddings_file=embeddings_file,
#     image_embeddings=image_embeddings,
#     save_embeddings=False,
# )
found_image_paths = ret.retrieve(
    query=prompt,
    encode_images_func=encode_images,
    encode_text_func=encode_prompt,
    embeddings_file=embeddings_file,
    save_embeddings=False,
)
t2 = time.time()
print(f"Time: {t2-t1}s")

# %% Display result in 3 columns
# json_request = defaultdict(list)
cols = st.columns(n_cols)
prompt_matched_image_paths = found_image_paths[:top_k]


def get_category(path):
    image_id = name(path)
    category = metadata[image_id]["semantic_category"]
    if category != "outerwear":
        category = category[:-1]
    return image_id, category


# Add chosen items into a dict to send request to outfit recommend api
ind_garment_retrieved = 0
ind_outfit = 0
for image_path in prompt_matched_image_paths:
    image_id, category = get_category(image_path)
    json_request[category].append(image_id)

# Add items in categories in which there is no item
# pprint(json_request)
for cate, items in json_request.items():
    count = 0
    if len(items) == 0:
        for image_path in found_image_paths[top_k:]:
            image_id, category = get_category(image_path)
            if category == cate:
                items.append(image_id)
                count += 1
            if count >= empty_cate_extras:
                break
# pprint(json_request)

# Send request to outfit recommend api
response = requests.post(
    url="http://127.0.0.1:3000/items/1/outfits_recommend_from_chosen/",
    json=json_request,
)

# Showcase api's reponse on web app
json_response = response.json()
outfit_recommends = json_response["outfit_recommend"]

for outfit in outfit_recommends:
    for cate in cates:
        garm_path = osp.join(image_dir, outfit[cate])
        with cols[int(ind_garment_retrieved % n_cols)]:
            st.image(garm_path, width=250)
            ind_garment_retrieved += 1

    ind_outfit += 1

    if ind_outfit >= 3:
        break
# %%
