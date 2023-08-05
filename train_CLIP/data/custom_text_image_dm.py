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

from data import process_text
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

        path2metadata = os.path.join(data_dir, "polyvore_item_metadata.json")

        self.metadatas = json.load(open(path2metadata, "r", encoding="UTF-8"))

        self.filter_description = filter_description

        self.resize_ratio = resize_ratio

        # self.image_transform = T.Compose(
        #     [
        #         T.Lambda(self.fix_img),
        #         T.RandomResizedCrop(
        #             image_size,
        #             scale=(self.resize_ratio, 1.0),
        #             ratio=(1.0, 1.0),
        #         ),
        #         T.ToTensor(),
        #         T.Normalize(
        #             (0.48145466, 0.4578275, 0.40821073),
        #             (0.26862954, 0.26130258, 0.27577711),
        #         ),
        #     ]
        # )
        self.image_transform = preprocess

        self.custom_tokenizer = custom_tokenizer

        self.clean_dataset()

    def __len__(self):
        """
        Return len of dataset
        """
        return len(self.image_fullpaths)

    def clean_dataset(self):
        """
        Cleaning the dataset:
            - Remove item without metadata
            - If filter_description, remove item without description in metadata
        """
        result = []
        for image_path in self.image_fullpaths:
            image_name = os.path.basename(image_path)
            item_id = os.path.splitext(image_name)[0]
            if item_id in self.metadatas:
                if self.filter_description:
                    description = self.metadatas[item_id]["description"]
                    if len(description) == 0:
                        continue
                result.append(image_path)
        self.image_fullpaths = result

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

    def process_metadata(self, metadata: dict):
        """
        Preprocess the metadata

        Args:
            metadata (dict): the metadata of the item

        Returns:
            str: the processed metadata
        """
        if self.filter_description:
            description = metadata.get("description", "").lower()
            processed_metadata = description
        else:
            url_name = metadata.get("url_name", "untitled").lower()
            if url_name == "untitled":
                url_name = ""

            description = metadata.get("description", "").lower()

            categories = metadata.get("catgeories", "")
            if type(categories) == list:
                categories = " ".join(categories).lower()

            title = metadata.get("title", "untitled").lower()
            if title == "untitled":
                title = ""

            related = metadata.get("related", "")
            if type(related) == list:
                related = " ".join(related).lower()

            semantic_category = metadata.get("semantic_category", "").lower()

            processed_metadata = (
                url_name
                + " "
                + description
                + " "
                + categories
                + " "
                + title
                + " "
                + related
                + " "
                + semantic_category
            )

        processed_metadata = process_text.remove_punctuation(processed_metadata)

        processed_metadata = process_text.remove_unwant_spaces(processed_metadata)

        return processed_metadata

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
        image_tensor = self.image_transform(image) if self.image_transform else image

        image_name = os.path.basename(image_fullpath)
        item_id = os.path.splitext(image_name)[0]
        item_metadata = self.process_metadata(self.metadatas[item_id])

        tokenized_text = (
            item_metadata if self.custom_tokenizer else tokenize(item_metadata)[0]
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
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
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
        self.train_dataset, self.val_dataset = random_split(dataset, [n_train, n_val])

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
            return torch.stack([row[0] for row in batch]), self.custom_tokenizer(
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
