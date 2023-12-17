# import os.path as osp
import os.path as osp
import streamlit as st
import requests

from tools import base64_to_image, load_json

# Some hyperparams
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
metadata_file = (
    "/home/dungmaster/Datasets/polyvore_outfits/polyvore_item_metadata.json"
)
metadata = load_json(metadata_file)

show_desc = False
n_cols = 5

name = lambda x: osp.basename(x).split(".")[0]

# Running the web app
title = "Fashion Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)

# Input search query
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "vacation outfit for men"

# Send request to outfit recommend api
response = requests.post(
    url="http://127.0.0.1:3000/fashion_retrieve",
    json={"text": prompt},
)

# Showcase api's reponse on web app
cols = st.columns(n_cols)

json_response = response.json()
garments_retrieved = json_response["garments_retrieved"]

for i, item_info in enumerate(garments_retrieved):
    img_name = name(item_info["image_name"])
    img_item = base64_to_image(item_info["image_base64"])
    info = metadata[img_name]

    if show_desc:
        desc = ""
        if len(info["description"]) != 0:
            desc = info["description"]
        elif len(info["title"]) != 0:
            desc = info["title"]
        elif len(info["url_name"]) != 0:
            desc = info["url_name"]

    with cols[int(i % n_cols)]:
        st.image(img_item, caption=(desc if show_desc else None), width=250)
