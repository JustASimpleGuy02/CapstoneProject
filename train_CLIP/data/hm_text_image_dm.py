from pathlib import Path
from random import randint, choice
import os
import json
from PIL import Image
import PIL
import argparse
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

from data.process_text import *
from clip import tokenize


class TextImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size=224,
        resize_ratio=0.75,
        shuffle=False,
        custom_tokenizer=False,
        filter_description=True,
        preprocess=None,
    ):
        """Create a text image dataset

        Args:
            data_dir (str): path to `polyvore_outfits` folder
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
           shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
            filter_description (bool, optional): Use only description as text. Defaults to True.
            preprocess (bool, optional): image transformation funcion
        """
        super().__init__()

        self.shuffle = shuffle

        path2images = os.path.join(data_dir, "images")

        image_files = os.listdir(path2images)

        self.image_fullpaths = [
            os.path.join(path2images, filename) for filename in image_files
        ]

        path2metadata = os.path.join(data_dir, "articles.csv")

        self.metadatas = pd.read_csv(path2metadata)

        self.resize_ratio = resize_ratio

        self.image_transform = preprocess

        self.custom_tokenizer = custom_tokenizer

        self.clean_dataset()

    def __len__(self):
        """
        Return len of dataset
        """
        return len(self.image_fullpaths)

    def clean_dataset(self):
        """ """
        # drop items that have the same description
        subset = self.metadatad.drop_duplicates("detail_desc").copy()

        # remove items of unkown category
        subset = subset[~subset["product_group_name"].isin(["Unknown"])]

        # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 tokens
        subset = subset[
            subset["detail_desc"].apply(lambda x: 4 < len(str(x).split()) < 40)
        ]

        # We also drop products types that do not occur very frequently in this subset of data
        most_frequent_product_types = [
            k
            for k, v in dict(
                Counter(subset["product_type_name"].tolist())
            ).items()
            if v > 10
        ]
        self.subset = subset[
            subset["product_type_name"].isin(most_frequent_product_types)
        ]
        del self.subset

    def fix_img(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):
        """
        Get one tuple of sample by index

        Args:
            index (int): index of the sample

        Returns:
            tuple: image, item_metadata, item_id
        """

        image_fullpath = self.image_fullpaths[index]
        image = Image.open(image_fullpath)
        image_tensor = (
            self.image_transform(image) if self.image_transform else image
        )

        image_name = os.path.basename(image_fullpath)
        item_id = os.path.splitext(image_name)[0]
        item_metadata = self.process_metadata(self.metadatas[item_id])

        tokenized_text = (
            item_metadata
            if self.custom_tokenizer
            else tokenize(item_metadata)[0]
        )

        return image_tensor, tokenized_text

    def get(self, index):
        """
        Get one tuple of sample by index

        Args:
            index (int): index of the sample

        Returns:
            tuple: image, item_metadata, item_id
        """

        image_fullpath = self.image_fullpaths[index]
        image = Image.open(image_fullpath)

        image_name = os.path.basename(image_fullpath)
        item_id = os.path.splitext(image_name)[0]
        item_metadata = self.process_metadata(self.metadatas[item_id])

        return image, item_metadata, image_name


class TextImageDataModule(LightningDataModule):
    def __init__(
        self,
        folder="data",
        batch_size=16,
        num_workers=8,
        image_size=224,
        resize_ratio=0.75,
        shuffle=False,
        custom_tokenizer=None,
        filter_description=True,
        preprocess=None,
    ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
            filter_description (bool, optional): Use only description as text. Defaults to True.
            preprocess (bool, optional): image transformation funcion
        """
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
        self.filter_description = filter_description
        self.preprocess = preprocess

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument(
            "--folder",
            type=str,
            required=True,
            help="directory of your training folder",
        )
        parser.add_argument("--batch_size", type=int, help="size of the batch")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="number of workers for the dataloaders",
        )
        parser.add_argument(
            "--image_size", type=int, default=224, help="size of the images"
        )
        parser.add_argument(
            "--resize_ratio",
            type=float,
            default=0.75,
            help="minimum size of images during random crop",
        )
        parser.add_argument(
            "--shuffle",
            type=bool,
            default=False,
            help="whether to use shuffling during sampling",
        )
        parser.add_argument(
            "--filter_description",
            type=bool,
            default=True,
            help="Use only description as text",
        )
        return parser

    def setup(self, stage=None):
        dataset = TextImageDataset(
            data_dir=self.folder,
            image_size=self.image_size,
            resize_ratio=self.resize_ratio,
            shuffle=self.shuffle,
            custom_tokenizer=not self.custom_tokenizer is None,
            filter_description=self.filter_description,
            preprocess=self.preprocess,
        )

        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(
            dataset, [n_train, n_val]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.dl_collate_fn,
        )

    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack(
                [row[1] for row in batch]
            )
        else:
            return torch.stack(
                [row[0] for row in batch]
            ), self.custom_tokenizer(
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
