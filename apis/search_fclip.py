from argparse import ArgumentParser
import os
import os.path as osp
from typing import List, Any
from glob import glob
import time

import numpy as np
import pandas as pd
from fashion_clip.fashion_clip import FashionCLIP
from tools import load_image, display_image_sets

get_filename = lambda x: osp.basename(x).split(".")[0]
norm = lambda x: np.linalg.norm(x, ord=2, axis=-1, keepdims=True)


def image_description(path: str, df: pd.DataFrame = None) -> str:
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


class FashionRetrieval:
    def __init__(
        self,
        model=FashionCLIP("fashion-clip"),
        image_embeddings: List[np.ndarray] = None,
        text_embeddings: List[np.ndarray] = None,
    ):
        self.model = model
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def retrieve(
        self,
        query: str,
        image_paths: List[str],
        encode_images_func: Any = None,
        encode_text_func: Any = None,
        embeddings_file: str = None,
        save_embeddings: bool = False,
        batch_size: int = 64,
    ):
        """Search top images predicted by the model according to query

        Args:
            query (str): input query to search for fashion item
            image_paths (List[str]): list of image paths to retrieve from
            encode_images_func (Any): function to encode images
            encode_text_func (Any): function to encode texts
            embeddings_file (str): absolute or relative path to image embeddings file
            save_embeddings (bool): whether to save embeddings into a file
            batch_size (int): number of items in a batch to process

        Returns:
            List[str]: list of result image paths sorted by rank
            np.ndarray: text embedding for later use
        """
        print("Embedding text...")
        text_embedding = (
            encode_text_func if encode_text_func else self.model.encode_text
        )([query], batch_size)[0]
        text_embedding = text_embedding[np.newaxis, ...]

        # if the input embedding file exist and user
        # does not ask to save the embedding
        if osp.exists(embeddings_file) and not save_embeddings:
            self.image_embeddings = np.loadtxt(embeddings_file)

        if self.image_embeddings is None:
            image_embeddings = (
                encode_images_func
                if encode_images_func
                else self.model.encode_images
            )(image_paths, batch_size=batch_size)
            self.image_embeddings = image_embeddings / norm(image_embeddings)

            if embeddings_file is not None and save_embeddings:
                print("Save image embeddings...")
                np.savetxt(embeddings_file, self.image_embeddings)

        print("Find matching fashion items...")
        cosine_sim = self.model._cosine_similarity(
            text_embedding, self.image_embeddings, normalize=True
        )
        inds = cosine_sim.argsort()[:, ::-1]
        inds = inds.squeeze().tolist()
        rank_image_paths = [image_paths[ind] for ind in inds]

        return rank_image_paths, text_embedding

    def classify(
        self,
        image: np.ndarray,
        classes: List[str],
        encode_image_func=None,
        encode_texts_func=None,
        embeddings_file: str = None,
        save_embeddings: bool = False,
        batch_size: int = 64,
    ):
        """Classify an image to the most matched text categories

        Args:
            image (np.ndarray): input image to classify
            classes (List[str]): list of classes to match image to
            encode_image_func (Any): function to encode images
            encode_texts_func (Any): function to encode texts
            embeddings_file (str): absolute or relative path to text embeddings file
            save_embeddings (bool): whether to save embeddings into a file
            batch_size (int): number of items in a batch to process

        Returns:
            (str): text category sorted by ranks
        """
        image_embedding = (
            encode_image_func if encode_image_func else self.model.encode_images
        )([image], batch_size)[0]
        image_embedding = image_embedding[np.newaxis, ...]

        if self.text_embeddings is None:
            text_embeddings = (
                encode_texts_func
                if encode_texts_func
                else self.model.encode_text
            )(classes, batch_size=batch_size)
            self.text_embeddings = text_embeddings / norm(text_embeddings)

            if embeddings_file is not None and save_embeddings:
                print("Save classes embeddings...")
                np.savetxt(embeddings_file, self.text_embeddings)

        print("Find matching class...")
        cosine_sim = self.model._cosine_similarity(
            image_embedding, self.text_embeddings, normalize=True
        )
        inds = cosine_sim.argsort()[:, ::-1]
        inds = inds.squeeze().tolist()
        rank_classes = [classes[ind] for ind in inds]

        return rank_classes


def parse_args():
    parser = ArgumentParser(description="Search fashion item using prompt")
    parser.add_argument("--image-dir", help="image data to search from")
    parser.add_argument("--prompt", help="prompt to search item")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        help="number of best items to choose",
        default=1,
    )
    parser.add_argument(
        "--embeddings-file", help="file to load and save image embeddings"
    )
    parser.add_argument(
        "-s",
        "--save-embeddings",
        action="store_true",
        help="whether to save image embeddings",
    )
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
    fclip = FashionCLIP("fashion-clip")
    image_paths = sorted(glob(osp.join(image_dir, "*.jpg")))
    assert (
        len(image_paths) > 0
    ), f"Directory {image_dir} does not have any .jpg images"

    image_embeddings = None

    if osp.isfile(embeddings_file):
        print(f"Load image embeddings from {embeddings_file}!")
        image_embeddings = np.loadtxt(embeddings_file)
    else:
        print(
            "Embedding file does not exist. Process the images and save embeddings!"
        )
        embeddings_dir = osp.dirname(embeddings_file)
        os.makedirs(embeddings_dir, exist_ok=True)
        save_embeddings = True

    t1 = time.time()
    found_image_paths, image_embeddings = search(
        fclip,
        prompt=prompt,
        image_paths=image_paths,
        embeddings_file=embeddings_file,
        image_embeddings=image_embeddings,
        top_k=top_k,
        save_embeddings=save_embeddings,
    )
    t2 = time.time()
    print(f"Time: {t2-t1}s")

    images = [load_image(path, backend="pillow") for path in found_image_paths]
    descs = [
        image_description(path, metadata_df) for path in found_image_paths
    ]

    display_image_sets(
        images=[images],
        set_titles=[prompt],
        descriptions=[descs],
        figsize=(20, 20),
        fontsize=8,
    )
    print("Done!!!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
