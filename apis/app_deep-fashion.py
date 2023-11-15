from glob import glob
import os.path as osp

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
image_dir = "/home/dungmaster/Datasets/Deep-Fashion/img"
top_k = 20

embeddings_file = osp.join(par_dir, "model_embeddings", "deep_fashion.txt")
image_embeddings = None
save_embeddings = True

if osp.exists(embeddings_file):
    image_embeddings = np.loadtxt(embeddings_file)
    save_embeddings = False

image_paths = glob(osp.join(image_dir, "*/*.jpg"))
print("Number of images:", len(image_paths))

ret = FashionRetrieval(
    image_paths=image_paths, image_embeddings=image_embeddings
)


# Main app
class TextInput(BaseModel):
    text: str


@app.post("/fashion_retrieve")
def retrieve_item(input_data: TextInput):
    processed_text = input_data.text.lower()
    found_image_paths, _ = ret.retrieve(
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
