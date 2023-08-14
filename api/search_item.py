import sys

sys.path.append("train_CLIP")
import os.path as osp
from typing import List
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
from clip import *

CHECKPOINT = "../training_logs/2023_08_12/epoch=11-step=10000.ckpt"
MODEL_NAME = "RN50"
CONFIG = "train_CLIP/clip/configs/RN.yaml"
DEVICE = "cuda"

model, preprocess = load(MODEL_NAME)
fclip = CLIPWrapper.load_from_checkpoint(
    CHECKPOINT,
    model_name=MODEL_NAME,
    model=model,
    minibatch_size=1,
).model.to(DEVICE)


def load_image(
    path_to_image: str, backend: str = "cv2", toRGB: bool = True
) -> np.ndarray:
    """Loading image from specied path

    Args:
        path_to_image (str): absolute paths to images
        toRGB (bool, optional): _description_. Defaults to True.
    """
    if backend == "cv2":
        image = cv2.imread(path_to_image)
        if toRGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == "pillow":
        image = np.array(Image.open(path_to_image))

    return image


def search(
    prompt: str,
    image_dir: str,
    image_embeddings: List[np.ndarray] = None,
    n_sample: int = None,
):
    """_summary_

    Args:
        prompt (str): input prompt to search for fashion item
        image_dir (List[np.ndarray]): directory to all fashion items images
        image_embeddings (List[np.ndarray], optional): embedding vectors of all the images. Defaults to None.
        n_sample (int): number of images to test if there are so many images

    Returns:
        np.ndarray: found image
        List[np.ndarray]: list of image embeddings for later use
    """
    image_paths = glob(osp.join(image_dir, "*.jpg"))

    if n_sample is None or n_sample > len(image_paths):
        n_sample = len(image_paths)

    # Encoding images
    if image_embeddings is None:
        image_embeddings = []
        for path in tqdm(image_paths[:n_sample]):
            image = Image.open(path)
            image_input = preprocess(image)[None, ...].cuda()
            with torch.no_grad():
                image_feature = fclip.encode_image(image_input).float()
            image_embeddings.append(image_feature)
        image_embeddings = torch.cat(image_embeddings)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    # Encoding prompt
    text_tokens = tokenize([prompt]).cuda()
    with torch.no_grad():
        text_embeddings = fclip.encode_text(text_tokens).float()

    # Calculating similarity matrix
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    similarity = text_embeddings.cpu().numpy() @ image_embeddings.cpu().numpy().T

    # Find most suitable image for the prompt
    id_of_matched_object = np.argmax(similarity)
    found_image_path = image_paths[id_of_matched_object]
    image = load_image(found_image_path)

    return image, image_embeddings


if __name__ == "__main__":
    image_dir = "data/demo/data_for_fashion_clip"
    
    image, image_embeddings = search("black shirt", image_dir, n_sample=1000)
    cv2.imshow("Found Image", image)
    cv2.waitKey(0)  
    
    image, _ = search(
        "brown shorts", image_dir=image_dir, image_embeddings=image_embeddings
    )
    cv2.imshow("Found Image", image)
    cv2.waitKey(0)


