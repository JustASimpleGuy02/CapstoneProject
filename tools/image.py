import random
from typing import Union, Tuple, List

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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

def display_image_with_desc_grid(
    img_desc_pairs: Union[List, Tuple],
    n_sample: int,
    n_rows: int,
):
    """Display grid of images with their descriptions

    Args:
        img_desc_pairs (Union[List, Tuple]): list or tuple of pairs of image-description
        n_sample (int): number of images to display
        n_rows (int): number of rows of the grid
    """
    n_cols = n_sample // n_rows

    f, ax = plt.subplots(n_rows, n_cols, figsize=(20,10))

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row*n_cols + col
            image, text = img_desc_pairs[idx]
            desc_list = text.split(' ')
            for j, elem in enumerate(desc_list):
                if j > 0 and j % 4 == 0:
                    desc_list[j] = desc_list[j] + '\n'
            text = ' '.join(desc_list)
            if n_rows == 1:
                ax[col].imshow(image)
                ax[col].set_xticks([], [])
                ax[col].set_yticks([], [])
                ax[col].grid(False)
                ax[col].set_xlabel(text, fontsize=10)
            else:
                ax[row, col].imshow(image)
                ax[row, col].set_xticks([], [])
                ax[row, col].set_yticks([], [])
                ax[row, col].grid(False)
                ax[row, col].set_xlabel(text, fontsize=10)

def display_image_sets(
    images: List[List[np.ndarray]],
    set_titles: List[str],
    figsize: Tuple[int, int] = (10, 20)
):
    """Display item sets with their titles

    Args:
        images List[List[np.ndarray]]: list of image paths to load and display
        set_titles (List[str]): list of titles accompanying each set 
        figsize (Tuple[int, int]): figsize to plot in matplotlib
    """
    n_rows = len(images)
    n_cols = max([len(set) for set in images])

    assert len(set_titles) == n_rows, "Number of titles must be equal to number of item sets"

    f, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        
    for row, set_items in enumerate(images):
        text = set_titles[row]
        desc_list = text.split(' ')
        for j, elem in enumerate(desc_list):
            if j > 0 and j % 4 == 0:
                desc_list[j] = desc_list[j] + '\n'
        text = ' '.join(desc_list)
        if n_rows == 1:
            ax[0].set_ylabel(text)
        else:
            ax[row, 0].set_ylabel(text)
        for col, image in enumerate(set_items):
            if n_rows == 1:
                ax[col].imshow(image)
                ax[col].set_xticks([], [])
                ax[col].set_yticks([], [])
                ax[col].grid(False)
            else:
                ax[row, col].imshow(image)
                ax[row, col].set_xticks([], [])
                ax[row, col].set_yticks([], [])
                ax[row, col].grid(False)
    plt.show()
