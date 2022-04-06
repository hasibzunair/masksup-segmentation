import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from PIL import Image
from torch.utils.data import Dataset


# Set random seed
np.random.seed(0)

"""Dataset classes"""

class ISIC2018_dataloader(Dataset):
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self._data_folder, 'ISIC2018_Task1-2_Training_Input')
        self._label_folder = os.path.join(self._data_folder, 'ISIC2018_Task1_Training_GroundTruth')
        self._images = sorted(glob.glob(self._input_folder + "/*.jpg"))
        self._labels = sorted(glob.glob(self._label_folder + "/*.png"))
        
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self._images, self._labels, 
                                                                            test_size=0.2, shuffle=False, random_state=0)
        
    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
            
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((256,256)),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        
        transforms_mask = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((256,256)),
                                             transforms.ToTensor()])
        
        image = transforms_image(image)
        mask = transforms_mask(mask)
        
        sample = {'image': image, 'mask': mask}
        return sample
    
    

class CVCClinicDB_dataloader(Dataset):
    """
    https://www.kaggle.com/balraj98/cvcclinicdb
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self._data_folder, 'Original')
        self._label_folder = os.path.join(self._data_folder, 'Ground Truth')
        self._images = sorted(glob.glob(self._input_folder + "/*.png"))
        self._labels = sorted(glob.glob(self._label_folder + "/*.png"))
        
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self._images, self._labels, 
                                                                            test_size=0.2, shuffle=False, random_state=0)
        
    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')
        
        # try this before centercrop transforms.Resize((256, 256))
        transforms_image = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((256,256)),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        
        transforms_mask = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((256,256)),
                                             transforms.ToTensor()])
        
        image = transforms_image(image)
        mask = transforms_mask(mask)
        
        sample = {'image': image, 'mask': mask}
        return sample
    
    
    

class ISIC2018Masked_dataloader(Dataset):
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self._data_folder, 'ISIC2018_Task1-2_Training_Input')
        self._scribbles_folder = os.path.join(self._data_folder, 'SCRIBBLES')
        self._images = sorted(glob.glob(self._input_folder + "/*.jpg"))
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))
        
        self.train_images, self.test_images, self.train_scribbles, self.test_scribbles = train_test_split(self._images, self._scribbles[:len(self._images)], 
                                                                            test_size=0.1, shuffle=False, random_state=0)
        
    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self._scribbles[np.random.randint(12000)] # pick randomly from scribbles
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_scribbles[idx]
            
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((192, 256)),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        
        transforms_mask = transforms.Compose([transforms.Resize((192, 256)),
                                             transforms.ToTensor()])
        
        image = transforms_image(image)
        mask = transforms_mask(mask)
        
        # Partial image
        partial_image1 = image * mask
        partial_image2 = image * (1 - mask)
        
        
        sample = {'image': image, 
                  'mask': mask, 
                  'partial_image1': partial_image1,
                  'partial_image2': partial_image2}
        return sample