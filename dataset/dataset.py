import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json

class PolyvoreDataset(Dataset):
    
    def __init__(self, data_dir, transform) -> None:
        """
        Init the dataset

        Args:
            data_dir (_type_): _description_
            transform (_type_): _description_
        """
        path2images = os.path.join(data_dir, "images")
        
        filenames = os.listdir(path2images)
        
        self.full_filenames = [os.path.join(path2images, filename)
                              for filename in filenames]
                
        path2metadata = os.path.join(data_dir, "polyvore_item_metadata.json")
        
        self.metadatas = json.load(open(path2metadata, 'r'))
        
        self.clean_dataset()
        
        self.transform = transform
        
    def clean_dataset(self):
        """
        Remove item without metadata
        """
        result = []
        for image_fullname in self.full_filenames:
            image_name = os.path.basename(image_fullname)
            item_id = os.path.splitext(image_name)[0]
            if item_id in self.metadatas:
                result.append(image_fullname)
        self.full_filenames = result
    
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
        image = self.transform(image)
        image_name = os.path.basename(image_fullname)
        item_id = os.path.splitext(image_name)[0]
        item_metadata = self.metadatas[item_id]
        
        return image, item_metadata, item_id