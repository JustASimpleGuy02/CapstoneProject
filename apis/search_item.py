import sys
sys.path.append("train_CLIP")
from argparse import ArgumentParser
import os.path as osp
from typing import List
from glob import glob
import random

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
from clip import *
from tools import load_checkpoint


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
    model,
    preprocess,
    prompt: str,
    image_paths: List[str],
    image_embeddings: List[np.ndarray] = None,
    n_sample: int = None,
    save_embeddings: bool = False,
):
    """_summary_

    Args:
        prompt (str): input prompt to search for fashion item
        image_paths (List[str]): list of absolute paths to item images
        image_embeddings (List[np.ndarray], optional): embedding vectors of all the images. Defaults to None.
        n_sample (int): number of images to test if there are so many images

    Returns:
        np.ndarray: found image
        List[np.ndarray]: list of image embeddings for later use
    """
    if n_sample is None or n_sample > len(image_paths):
        n_sample = len(image_paths)

    # Encoding images
    if image_embeddings is None:
        image_embeddings = []
        for path in tqdm(image_paths[:n_sample]):
            image = Image.open(path)
            image_input = preprocess(image)[None, ...].cuda()
            with torch.no_grad():
                image_feature = model.encode_image(image_input).float()
            image_embeddings.append(image_feature)
        image_embeddings = torch.cat(image_embeddings)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings.cpu().numpy()

    if save_embeddings:
        np.savetxt("image_embeddings_demo.txt", image_embeddings)

    # Encoding prompt
    text_tokens = tokenize([prompt]).cuda()
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens).float()
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.cpu().numpy()

    # Calculating cosine similarity matrix
    similarity = text_embeddings @ image_embeddings.T

    # Find most suitable image for the prompt
    id_of_matched_object = np.argmax(similarity)
    found_image_path = image_paths[id_of_matched_object]
    image = load_image(found_image_path)

    return image, image_embeddings

def parse_args():
    parser = ArgumentParser(description="Search fashion item using prompt")
    parser.add_argument("prompt", help="prompt to search item")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    prompt = args.prompt 
    
    image_dir = "data/demo/data_for_fashion_clip"
    model_path = "../training_logs/2023_08_14/epoch=50-step=45951.ckpt"
    # model_path = "../training_logs/2023_08_12/epoch=11-step=10000.ckpt"
    model_name = "RN50"

    model, preprocess = load_checkpoint(model_path, model_name)
    image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))
    image_embeddings = np.loadtxt("image_embeddings_demo.txt")

    image, image_embeddings = search(
        model,
        preprocess,
        prompt=prompt,
        image_paths=image_paths,
        image_embeddings=image_embeddings,
    )
    image = cv2.resize(image, (512, 512))
    cv2.imshow("Found Image", image)
    cv2.waitKey(0)

    