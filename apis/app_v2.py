import os.path as osp
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import numpy as np
from PIL import Image

from .search_item_fashion_clip import FashionRetrieval
from tools import image_to_base64

app = FastAPI()

# Load hyperparams
project_dir = "/home/dungmaster/Projects/Machine Learning"
par_dir = osp.join(
    project_dir, "HangerAI_outfits_recommendation_system/CapstoneProject"
)
image_dir = "/home/dungmaster/Datasets/polyvore_outfits/images"
top_k = 20

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


def load_image_files(hfile):
    pkl_file = open(hfile, "rb")
    hashes_polyvore = pickle.load(pkl_file)[1]
    image_names = list(hashes_polyvore.keys())
    image_paths = [osp.join(image_dir, name) for name in image_names]
    return image_paths


image_paths = load_image_files(hashes_file)
image_embeddings = np.loadtxt(embeddings_file)

ret = FashionRetrieval(
    image_paths=image_paths, image_embeddings=image_embeddings
)


# Main app
class TextInput(BaseModel):
    text: str


@app.post("/fashion_retrieve")
def retrieve_item(input_data: TextInput):
    processed_text = input_data.text.lower()
    found_image_paths = ret.retrieve(query=processed_text)[:top_k]

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
