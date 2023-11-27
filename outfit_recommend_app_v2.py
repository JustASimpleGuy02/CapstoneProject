import sys
import os.path as osp
import time
import requests
from pprint import pprint

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tools import display_centered_text, load_json

# Some hyperparams
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
# cates = ["outerwear", "full-body", "top", "bottom", "bag", "footwear", "accessory"]
cates = ["top", "bottom", "bag", "outerwear", "shoe"]

metadata_file = "/home/dungmaster/Datasets/polyvore_outfits/polyvore_item_metadata.json"
metadata = load_json(metadata_file)

n_cols = len(cates)
n_outfits = 4
show_desc = False
debug = True
table_style = False

name = lambda x: osp.basename(x).split(".")[0]

# Running the web app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)

# Input search query
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "skiing outfit for women"

# Send request to outfit recommend api
t1 = time.time()
response = requests.post(
    url="http://127.0.0.1:3000/items/1/outfits_recommend_from_prompt/",
    json={"text": prompt},
)
t2 = time.time()
print(f"Time: {(t2-t1):.3f}s")

# Showcase api's reponse on web app
cols = st.columns(n_cols, )
print(n_cols)

json_response = response.json()
pprint(json_response)

first_item_cates = json_response["first_item_cate"]
outfit_recommends = json_response["outfit_recommend"]

if table_style:
    outfit_df = pd.DataFrame(outfit_recommends)
    image_columns = {
        c: st.column_config.ImageColumn(
            help="Garment screenshot"
        )
        for c in cates
    }

    st.data_editor(
        outfit_df,
        width=1920,
        column_config=image_columns,
    )
else:
    ind_garment_retrieved = 0
    ind_outfit = 0

    for ind, outfit in enumerate(outfit_recommends):
        first_item_cate = first_item_cates[ind]
        print(first_item_cate)

        for cate in cates:
            garm_path = outfit[cate]

            if show_desc:
                img_name = name(osp.basename(garm_path))
                info = metadata[img_name]

                desc = ""
                if len(info["description"]) != 0:
                    desc = info["description"]
                elif len(info["title"]) != 0:
                    desc = info["title"]
                elif len(info["url_name"]) != 0:
                    desc = info["url_name"]

            ind_col = int(ind_garment_retrieved % n_cols)
            col = cols[ind_col]

            with col:
                if ind_outfit == 0:
                    st.header(cate)

                image = Image.open(osp.join(image_dir, garm_path))

                if cate == first_item_cate and debug:
                    image = ImageOps.expand(image, border=5, fill="yellow")

                st.image(
                    image, caption=(desc if show_desc else None), width=250
                )
                ind_garment_retrieved += 1

        ind_outfit += 1

        if ind_outfit >= n_outfits:
            break
