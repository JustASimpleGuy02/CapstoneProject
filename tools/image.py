import random
from typing import Union, Tuple, List
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def split_text_into_lines(text: str):
    """Split text into multiple lines to display with image

    Args:
        text (str): input text

    Returns
        (str): result text
    """
    desc_list = text.split(" ")
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 2 == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = " ".join(desc_list)
    return text


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


def base64_to_image(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Create an Image object from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image


def display_image_with_desc_grid(
    img_desc_pairs: Union[List, Tuple],
    n_sample: int,
    n_rows: int,
    figsize: Tuple[int, int] = (10, 20),
    fontsize: int = 10,
):
    """Display grid of images with their descriptions

    Args:
        img_desc_pairs (Union[List, Tuple]): list or tuple of pairs of image-description
        n_sample (int): number of images to display
        n_rows (int): number of rows of the grid
        figsize (Tuple[int, int]): figsize to plot in matplotlib
        fontsize (int): font size of each text description of image
    """
    n_cols = n_sample // n_rows

    figs, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            image, text = img_desc_pairs[idx]
            desc_list = text.split(" ")
            for j, elem in enumerate(desc_list):
                if j > 0 and j % 4 == 0:
                    desc_list[j] = desc_list[j] + "\n"
            text = " ".join(desc_list)
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            ax.imshow(image)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.grid(False)
            ax.set_xlabel(text, fontsize=fontsize)

    plt.show()


def display_image_sets(
    images: List[List[np.ndarray]],
    set_titles: List[str],
    descriptions: List[List[str]] = None,
    figsize: Tuple[int, int] = (10, 20),
    fontsize: int = 10,
):
    """Display item sets with their titles

    Args:
        images (List[List[np.ndarray]]): list of images to load and display
        set_titles (List[str]): list of titles accompanying each set
        descriptions (List[List[str]]): list of description of each item, default None
        figsize (Tuple[int, int]): figsize to plot in matplotlib
        fontsize (int): font size of each text description of image
    """
    n_rows = len(images)
    n_cols = max([len(set) for set in images])

    assert (
        len(set_titles) == n_rows
    ), "Number of titles must be equal to number of item sets"

    figs, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    figs.subplots_adjust(bottom=0.5, wspace=1.0)

    for row, set_items in enumerate(images):
        text = set_titles[row]
        text = split_text_into_lines(text)
        if n_rows == 1:
            axes[0].set_ylabel(text, fontsize=fontsize)
        else:
            axes[row, 0].set_ylabel(text, fontsize=fontsize)
        for col, image in enumerate(set_items):
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            if descriptions is not None:
                desc = descriptions[row][col]
                desc = split_text_into_lines(desc)
                ax.set_xlabel(desc, fontsize=fontsize)

            ax.imshow(image)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.grid(False)

    plt.show()
