import os.path as osp
import streamlit as st
import requests

from tools import load_json

# Some hyperparams
image_dir = "/home/dungmaster/Datasets/Deep-Fashion/img"

show_desc = True
n_cols = 5

name = lambda x: osp.basename(x).split(".")[0]

# Running the web app
title = "Fashion Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)

# Input search query
prompt = st.text_input("Search: ")
# if len(prompt) == 0:
#     prompt = "vacation outfit for men"

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
    img = name(item_info["image_name"])

    with cols[int(i % n_cols)]:
        st.image(img, width=250)
