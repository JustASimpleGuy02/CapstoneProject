import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import dataset.process_text as process_text
import pickle

class PolyvoreDataset(Dataset):
    
    def __init__(self, data_dir, transform = None, filter_description = True) -> None:
        """
        Initialize the dataset

        Args:
            data_dir (str): path to polyvore_outfits folder
            transform (transform): the transform apply to image. Default to None
        """
        path2images = os.path.join(data_dir, "images")
        
        filenames = os.listdir(path2images)
        
        self.full_filenames = [os.path.join(path2images, filename)
                              for filename in filenames]
                
        path2metadata = os.path.join(data_dir, "polyvore_item_metadata.json")
        
        self.metadatas = json.load(open(path2metadata, 'r', encoding='UTF-8'))
        
        self.filter_description = filter_description
        
        self.clean_dataset()
        
        self.transform = transform
        
    def clean_dataset(self):
        """
        Cleaning the dataset:
            - Remove item without metadata
            - If filter_description, remove item without description in metadata
        """
        result = []
        for image_fullname in self.full_filenames:
            image_name = os.path.basename(image_fullname)
            item_id = os.path.splitext(image_name)[0]
            if item_id in self.metadatas:
                if self.filter_description:
                    description = self.metadatas[item_id]["description"]
                    if len(description) == 0:
                        continue
                result.append(image_fullname)
        self.full_filenames = result
    
    def process_metadata(self, metadata: dict):
        """
        Preprocess the metadata

        Args:
            metadata (str): the metadata of the item

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
            
            processed_metadata = url_name + " " + description + " " + \
                categories + " " + title + " " + related + " " + semantic_category
        
        processed_metadata = process_text.remove_punctuation(processed_metadata)
        
        processed_metadata = process_text.remove_unwant_spaces(processed_metadata)
        
        return processed_metadata
    
    def __len__(self):
        """
        Return len of dataset
        """
        return len(self.full_filenames)
    
    def __getitem__(self, index):
        """
        Get one tuple of sample by index

        Args:
            index (int): index of the sample

        Returns:
            tuple: image, item_metadata, item_id
        """
        image_fullname = self.full_filenames[index]
        image = Image.open(image_fullname)
        if self.transform is not None:
            image = self.transform(image)
        image_name = os.path.basename(image_fullname)
        item_id = os.path.splitext(image_name)[0]
        item_metadata = self.process_metadata(self.metadatas[item_id])
        
        return image, item_metadata, image_name

class PolyvoreFashionHashDataset(Dataset):
    
    def __init__(self, data_dir, transform = None) -> None:
        """
        Initialize the dataset

        Args:
            data_dir (str): path to polyvore_outfits folder
            transform (transform): the transform apply to image. Default to None
        """
        path2images = os.path.join(data_dir, "images", "291x291")
        
        filenames = os.listdir(path2images)
        
        self.full_filenames = [os.path.join(path2images, filename)
                              for filename in filenames]
        
        path2vector = os.path.join(data_dir, "sentence_vector", "semantic.pkl")
        
        path2metadata = os.path.join(data_dir, "fashion_items.pickle")
        
        with open(path2vector, 'rb') as f:
            self.vector_dict = pickle.load(f)
        
        with open(path2metadata, 'rb') as f:
            self.metadata_dict = pickle.load(f)
    
        self.transform = transform
    
    def __len__(self):
        """
        Return len of dataset
        """
        return len(self.full_filenames)
    
    def __getitem__(self, index):
        image_fullname = self.full_filenames[index]
        image = Image.open(image_fullname)
        if self.transform is not None:
            image = self.transform(image)
        image_name = os.path.basename(image_fullname)
        semantic_vector = self.vector_dict[image_name]
        item_metadata = self.metadata_dict[image_name]
        
        return image, semantic_vector, item_metadata, image_name