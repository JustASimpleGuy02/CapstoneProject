import os.path as osp
import time
import requests
import streamlit as st

import sys
sys.path += [".."]
from tools import load_json

# Some hyperparams
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
cates = ["top", "bottom", "bag", "outerwear", "shoe"]

metadata_file = (
    "/home/dungmaster/Datasets/polyvore_outfits/polyvore_item_metad\
ata.json"
)
metadata = load_json(metadata_file)

n_cols = len(cates)
n_outfits = 4
show_desc = False
n_cols = 5

name = lambda x: osp.basename(x).split(".")[0]

# Running the web app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)

# Input search query
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "gym outfit for men"

# Send request to outfit recommend api
t1 = time.time()
response = requests.post(
    url="http://127.0.0.1:3000/items/1/outfits_recommend_from_prompt/",
    json={"text": prompt},
)
t2 = time.time()
print(f"Time: {(t2-t1):.3f}s")

# Showcase api's reponse on web app
cols = st.columns(n_cols)

json_response = response.json()
outfit_recommends = json_response["outfit_recommend"]

ind_garment_retrieved = 0
ind_outfit = 0
for outfit in outfit_recommends:
    for cate in cates:
        garm_path = osp.join(image_dir, outfit[cate])

        img_name = name(osp.basename(garm_path))
        info = metadata[img_name]

        if show_desc:
            desc = ""
            if len(info["description"]) != 0:
                desc = info["description"]
            elif len(info["title"]) != 0:
                desc = info["title"]
            elif len(info["url_name"]) != 0:
                desc = info["url_name"]
        
        with cols[int(ind_garment_retrieved % n_cols)]:
            st.image(garm_path,
                     caption=(desc if show_desc else None),
                     width=250)
            ind_garment_retrieved += 1

    ind_outfit += 1

    if ind_outfit >= n_outfits:
        break
