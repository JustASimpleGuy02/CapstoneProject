from typing import Union, Tuple, List
import random
import cv2
import matplotlib.pyplot as plt

def load_image(
    path: str,
    rgb: bool = True
):
    """Load image using opencv-library
    
    Args:
    path (str): relative or absolute path to the image
    rgb (bool): whether to convert image to rgb
    
    Returns:
    (np.ndarray): the loaded image
    """
    image = cv2.imread(path)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
