from os import environ

environ["OMP_NUM_THREADS"] = "1"
import os
import os.path as osp
import time
from glob import glob
from PyQt5.QtCore import pyqtSignal
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import cv2
from PIL import Image

fclip = FashionCLIP("fashion-clip")

image_dir = "data/demo/data_for_fashion_clip"

embeddings_file = "model_embeddings/fashion-clip/image_embeddings_demo.txt"

image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))

image_embeddings = np.loadtxt(embeddings_file)

top_k = 5


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


def search(prompt: str, search_done, n_sample: int = -1):
    """_summary_

    Args:
        prompt (str): input prompt to search for fashion item
        search_done: signal emit when function complete
        n_sample (int): number of images to test if there are so many images
    """
    t1 = time.time()
    text_embedding = fclip.encode_text([prompt], 32)[0]

    similarity = text_embedding @ image_embeddings.T
    if len(similarity.shape) > 1:
        similarity = np.squeeze(similarity)

    matched_inds = np.argpartition(similarity, -top_k)[-top_k:]
    sorted_inds = matched_inds[np.argsort(similarity[matched_inds])][::-1]
    found_image_paths = [image_paths[ind] for ind in sorted_inds]

    t2 = time.time()

    print(t2 - t1)

    #    id_of_matched_object = np.argmax(text_embedding.dot(image_embeddings.T))
    #    found_image_name = image_names[id_of_matched_object]

    images = [load_image(path, backend="pillow") for path in found_image_paths]

    search_done.emit(images)


#    images_result = np.hstack(images)

#    search_done.emit(images_result)


def embedding(image_dir: str, n_sample: int = 0):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if n_sample > 0:
        paths = image_paths[:n_sample]
        embeddings = fclip.encode_images(paths, batch_size=32)
    else:
        paths = image_paths
        embeddings = fclip.encode_images(paths, batch_size=32)
    return paths, embeddings
