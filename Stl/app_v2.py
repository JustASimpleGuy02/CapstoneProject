import os.path as osp
import streamlit as st
import requests

# Some hyperparams
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
cates = ["top", "bottom", "bag", "outerwear", "shoe"]
n_cols = len(cates)

# Running the web app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)

# Input search query
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "wedding suit for men"

# Send request to outfit recommend api
response = requests.post(
    url="http://127.0.0.1:3000/items/1/outfits_recommend_from_prompt/",
    json={"text": prompt},
)

# Showcase api's reponse on web app
cols = st.columns(n_cols)

json_response = response.json()
outfit_recommends = json_response["outfit_recommend"]

ind_garment_retrieved = 0
ind_outfit = 0
for outfit in outfit_recommends:
    for cate in cates:
        garm_path = osp.join(image_dir, outfit[cate])
        with cols[int(ind_garment_retrieved % n_cols)]:
            st.image(garm_path, width=250)
            ind_garment_retrieved += 1

    ind_outfit += 1

    if ind_outfit >= 3:
        break
