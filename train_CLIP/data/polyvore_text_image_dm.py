from pathlib import Path
from random import randint, choice
import os
import os.path as osp
import json
import argparse

import polars as pl
from PIL import Image
import PIL
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
        csv_metadata: str,
        shuffle=False,
        custom_tokenizer=False,
        image_transform=None,
    ):
        """Create a text image dataset

        Args:
            data_dir (str): path to `polyvore_outfits` folder
            csv_metadata (str): path to csv file whose content is every image with its description
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
            image_transform (bool, optional): image transformation funcion
        """
        super().__init__()

        self.shuffle = shuffle

        self.image_dir = os.path.join(data_dir, "images")

        self.metadata = pl.read_csv(csv_metadata)

        self.image_transform = image_transform

        self.custom_tokenizer = custom_tokenizer

    def __len__(self):
        """
        Return len of dataset
        """
        return len(self.metadata)

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
        text, image_name = self.metadata[index]
        image_path = osp.join(self.image_dir, image_name.item())
        image = Image.open(image_path)
        image_tensor = (
            self.image_transform(image) if self.image_transform else image
        )

        text = text.item()
        tokenized_text = text if self.custom_tokenizer else tokenize(text)[0]

        return image_tensor, tokenized_text

    def get(self, index):
        """
        Get one tuple of sample by index

        Args:
            index (int): index of the sample

        Returns:
            tuple: image, text, item_id
        """
        text, image_name = self.metadata[index]
        image_path = osp.join(image_dir, image_name.item())
        image = Image.open(image_path)
        text = text.item()

        return image, text, image_name


class TextImageDataModule(LightningDataModule):
    def __init__(
        self,
        folder="data",
        csv="data/polyvore_img_desc.csv",
        batch_size=16,
        num_workers=8,
        shuffle=False,
        custom_tokenizer=None,
        image_transform=None,
    ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
            image_transform (bool, optional): image transformation funcion
        """
        super().__init__()
        self.folder = folder
        self.csv = csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
        self.image_transform = image_transform

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
        parser.add_argument(
            "--csv",
            type=str,
            required=True,
            help="path to metadata csv file",
        )
        parser.add_argument("--batch_size", type=int, help="size of the batch")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="number of workers for the dataloaders",
        )
        parser.add_argument(
            "--shuffle",
            type=bool,
            default=False,
            help="whether to use shuffling during sampling",
        )
        return parser

    def setup(self, stage=None):
        dataset = TextImageDataset(
            data_dir=self.folder,
            csv_metadata=self.csv,
            shuffle=self.shuffle,
            custom_tokenizer=not self.custom_tokenizer is None,
            image_transform=self.image_transform,
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
