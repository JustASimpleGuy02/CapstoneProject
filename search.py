import os.path as osp
from typing import List
from glob import glob

from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import cv2
from PIL import Image

fclip = FashionCLIP('fashion-clip')

def load_image(path_to_image: str, backend: str = 'cv2', toRGB: bool = True) -> np.ndarray:
    """Loading image from specied path

    Args:
        path_to_image (str): absolute paths to images
        toRGB (bool, optional): _description_. Defaults to True.
    """
    if backend == 'cv2':
        image = cv2.imread(path_to_image)
        if toRGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == 'pillow':
        image = np.array(Image.open(path_to_image))

    return image

def search(prompt: str,
           image_dir: str,
           image_embeddings: List[np.ndarray] = None,
           n_sample: int = -1):
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
    image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))
    text_embedding = fclip.encode_text([prompt], 32)[0]

    if image_embeddings is None:
        image_embeddings = fclip.encode_images(image_paths[:n_sample], batch_size=32)

    id_of_matched_object = np.argmax(text_embedding.dot(image_embeddings.T))
    found_image_path = image_paths[id_of_matched_object]

    image = load_image(found_image_path)

    return image, image_embeddings

if __name__ == "__main__":
    image_dir = "fashion-items-dataset"
    image, image_embeddings = search("black shirt", image_dir, n_sample=200)
    image, _ = search("brown shorts", image_dir=image_dir, image_embeddings=image_embeddings)