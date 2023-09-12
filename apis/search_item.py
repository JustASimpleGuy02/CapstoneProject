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
# from clip import *
from tools import load_model
from tools import *

def search(
    model,
    preprocess,
    prompt: str,
    image_paths: List[str],
    embeddings_file: str = None,
    image_embeddings: List[np.ndarray] = None,
    n_sample: int = None,
    top_k: int = 5,
    save_embeddings: bool = False,
):
    """Search top images predicted by the model according to prompt

    Args:
        prompt (str): input prompt to search for fashion item
        image_paths (List[str]): list of absolute paths to item images
        embeddings_file (str): absolute or relative path to image embeddings file
        image_embeddings (List[np.ndarray], optional): embedding vectors of all the images. Defaults to None.
        top_k (int): number of best images to choose
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
        print("Save image embeddings...")
        np.savetxt(embeddings_file, image_embeddings)

    # Encoding prompt
    text_tokens = tokenize([prompt]).cuda()
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens).float()
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.cpu().numpy()

    # Calculating cosine similarity matrix
    similarity = text_embeddings @ image_embeddings.T
    if len(similarity.shape) > 1:
        similarity = np.squeeze(similarity)

    # Find most suitable image for the prompt
    matched_inds = np.argpartition(similarity, -top_k)[-top_k:]
    sorted_inds = matched_inds[np.argsort(similarity[matched_inds])][::-1]
    found_image_paths = [image_paths[ind] for ind in sorted_inds]

    return found_image_paths, image_embeddings

def parse_args():
    parser = ArgumentParser(description="Search fashion item using prompt")
    parser.add_argument("--image-dir", help="image data to search from")
    parser.add_argument("--model-name", help="name of the model")
    parser.add_argument("--model-path", help="absolute or relative path to model")
    parser.add_argument("--prompt", help="prompt to search item")
    parser.add_argument("-k", "--top-k", type=int, help="number of best items to choose", default=1)
    parser.add_argument("--embeddings-file", help="file to load and save image embeddings")
    parser.add_argument("-s", "--save-embeddings", action="store_true", help="whether to save image embeddings")        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    prompt = args.prompt 
    image_dir = args.image_dir
    model_name = args.model_name
    model_path = args.model_path
    top_k = args.top_k
    embeddings_file = args.embeddings_file
    save_embeddings = args.save_embeddings

    print("Loading model...")
    model, preprocess = load_model(model_path, model_name)
    image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))
    assert len(image_paths) > 0, f"Directory {image_dir} does not have any .jpg images"

    image_embeddings = None

    if osp.isfile(embeddings_file):
        print(f"Load image embeddings from {embeddings_file}!")
        image_embeddings = np.loadtxt(embeddings_file)
    else:
        print("Embedding file does not exist. Process the images and save embeddings!")
        save_embeddings = True

    found_image_paths, image_embeddings = search(
        model,
        preprocess,
        prompt=prompt,
        image_paths=image_paths,
        embeddings_file=embeddings_file,
        image_embeddings=image_embeddings,
        top_k = top_k,
        save_embeddings=save_embeddings
    )
    images = [load_image(path, toRGB=False) for path in found_image_paths]
    display_image_sets(
        images=[images],
        set_titles=[prompt]
    )
    print("Done!!!")
    
