import os.path as osp
import json

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import numpy as np
import pandas as pd
from PIL import Image

from .search_fclip import FashionRetrieval
from tools import image_to_base64
from icecream import ic

app = FastAPI()

# Load hyperparams
project_dir = "/home/dungmaster/Projects/Machine Learning"
par_dir = osp.join(
    project_dir, "HangerAI_outfits_recommendation_system/CapstoneProject"
)
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "sample_images")
item2cate_fpath = osp.join(data_dir, "item_cate_502.json")

item_cate_map = json.load(open(item2cate_fpath, 'r'))
top_k = 20
search_category = "bag"

image_paths = [osp.join(image_dir, img_fn) for img_fn in item_cate_map[search_category]]
embeddings_file = osp.join(par_dir, "model_embeddings", "polyvore", f"polyvore_{search_category}_502.txt")
save_embeddings = True

if osp.exists(embeddings_file):
    image_embeddings = np.loadtxt(embeddings_file)
    save_embeddings = False
else:
    image_embeddings = None

ret = FashionRetrieval(
    image_embeddings=image_embeddings
)


# Main app
class TextInput(BaseModel):
    text: str


@app.post("/fashion_retrieve")
def retrieve_item(input_data: TextInput):
    processed_text = input_data.text.lower()
    found_image_paths, _ = ret.retrieve(
        image_paths=image_paths,
        query=processed_text,
        embeddings_file=embeddings_file,
        save_embeddings=save_embeddings
    )

    json_obj = {
        "garments_retrieved": [
            {
                "image_name": osp.basename(path),
                "image_path": path,
                "image_base64": image_to_base64(Image.open(path)),
            }
            for path in found_image_paths
        ]
    }
    return json_obj


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
