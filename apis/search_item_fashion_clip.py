import sys
sys.path.append("train_CLIP")
from argparse import ArgumentParser
import os
import os.path as osp
from typing import List
from glob import glob
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from fashion_clip.fashion_clip import FashionCLIP
from tools import *

get_filename = lambda x: osp.basename(x).split('.')[0]

def image_description(
    path: str,
    df: pd.DataFrame = None
) -> str:
    """Get description of image from image path
    
    Args:
        path (str): path to image item
        df (pd.DataFrame): metadata dataframe 
    Returns:
        str: string description of the item
    """
    item_id = int(get_filename(path))
    desc = df[df["article_id"] == item_id]["detail_desc"]
    return desc.item()
    
def search(
    model,
    prompt: str,
    image_paths: List[str],
    embeddings_file: str = None,
    image_embeddings: List[np.ndarray] = None,
    n_sample: int = None,
    top_k: int = 5,
    save_embeddings: bool = False,
    batch_size: int = 64
):
    """Search top images predicted by the model according to prompt

    Args:
        model (Any): the pretrained model
        prompt (str): input prompt to search for fashion item
        image_paths (List[str]): list of absolute paths to item images
        embeddings_file (str): absolute or relative path to image embeddings file
        image_embeddings (List[np.ndarray], optional): embedding vectors of all the images. Defaults to None.
        top_k (int): number of best images to choose
        n_sample (int): number of images to test if there are so many images
        batch_size (int): number of items in a batch to process

    Returns:
        List[str]: list of result image paths
        List[np.ndarray]: list of image embeddings for later use
    """
    text_embedding = model.encode_text([prompt], batch_size)[0]

    if n_sample is None or n_sample > len(image_paths):
        n_sample = len(image_paths)

    if image_embeddings is None:
        image_embeddings = model.encode_images(image_paths[:n_sample], batch_size=batch_size)

    if save_embeddings:
        print("Save image embeddings...")
        np.savetxt(embeddings_file, image_embeddings)
        
    similarity = text_embedding.dot(image_embeddings.T)

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
    parser.add_argument("--prompt", help="prompt to search item")
    parser.add_argument("-k", "--top-k", type=int, help="number of best items to choose", default=1)
    parser.add_argument("--embeddings-file", help="file to load and save image embeddings")
    parser.add_argument("-s", "--save-embeddings", action="store_true", help="whether to save image embeddings")        
    args = parser.parse_args()
    return args

def main(args):
    metadata_df = pd.read_csv("data/hm/articles.csv")
    prompt = args.prompt 
    image_dir = args.image_dir
    top_k = args.top_k
    embeddings_file = args.embeddings_file
    save_embeddings = args.save_embeddings

    print("Loading model...")
    fclip = FashionCLIP('fashion-clip')
    image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))
    assert len(image_paths) > 0, f"Directory {image_dir} does not have any .jpg images"

    image_embeddings = None

    if osp.isfile(embeddings_file):
        print(f"Load image embeddings from {embeddings_file}!")
        image_embeddings = np.loadtxt(embeddings_file)
    else:
        print("Embedding file does not exist. Process the images and save embeddings!")
        embeddings_dir = osp.dirname(embeddings_file)
        os.makedirs(embeddings_dir, exist_ok=True)
        save_embeddings = True

    t1 = time.time()
    #TODO: plot each result image with its description
    found_image_paths, image_embeddings = search(
        fclip,
        prompt=prompt,
        image_paths=image_paths,
        embeddings_file=embeddings_file,
        image_embeddings=image_embeddings,
        top_k = top_k,
        save_embeddings=save_embeddings
    )
    t2 = time.time()
    print(f"Time: {t2-t1}s")
    
    images = [load_image(path, backend="pillow") for path in found_image_paths]
    descs = [image_description(path, metadata_df) for path in found_image_paths]
    
    display_image_sets(
        images=[images],
        set_titles=[prompt],
        descriptions=[descs],
        figsize=(20, 20),
        fontsize=8
    )
    print("Done!!!")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
