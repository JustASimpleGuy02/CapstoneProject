import os
from glob import glob
from PyQt5.QtCore import pyqtSignal
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import cv2
from PIL import Image

fclip = FashionCLIP('fashion-clip')

DATA_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data"))

img_dir = os.path.join(DATA_PATH, "demo", "data_for_fashion_clip")

img_name_dir = os.path.join(DATA_PATH, "demo", "preprocess_data", "img_names.txt")

embedding_dir = os.path.join(DATA_PATH, "demo", "preprocess_data", "embedding.npy")

with open(img_name_dir, 'r') as f:
    lines = f.readlines()

image_names = [line.strip() for line in lines]
image_embeddings = np.load(embedding_dir)

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

def search_image(prompt: str,
             n_sample: int = -1):
    """_summary_

    Args:
        prompt (str): input prompt to search for fashion item
        search_done: signal emit when function complete
        n_sample (int): number of images to test if there are so many images
    """
    
    text_embedding = fclip.encode_text([prompt], 32)[0]


    id_of_matched_object = np.argmax(text_embedding.dot(image_embeddings.T))
    found_image_name = image_names[id_of_matched_object]

    image_path = os.path.join(img_dir, found_image_name)

    image = load_image(image_path)
    
    image_1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return os.path.basename(image_path)

def embedding(image_dir: str, n_sample: int = 0):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if n_sample > 0:
        paths = image_paths[:n_sample]
        embeddings = fclip.encode_images(paths, batch_size=32)
    else:
        paths = image_paths
        embeddings = fclip.encode_images(paths, batch_size=32)
    return paths, embeddings