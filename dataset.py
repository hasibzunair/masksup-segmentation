import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import torch

from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset

"""Dataset classes"""

class ISIC2018_dataloader(Dataset):
    """
    ISIC 2018 data loader with Irregular Masks Dataset (https://arxiv.org/abs/1804.07723)
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self._data_folder, 'ISIC2018_Task1-2_Training_Input')
        self._label_folder = os.path.join(self._data_folder, 'ISIC2018_Task1_Training_GroundTruth')
        self._scribbles_folder = os.path.join(self._data_folder, 'SCRIBBLES')
        self._images = sorted(glob.glob(self._input_folder + "/*.jpg"))
        self._labels = sorted(glob.glob(self._label_folder + "/*.png"))
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))[:1000] # For heavy masking [::-1]
        
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self._images, 
                                                                                                    self._labels,
                                                                                                    test_size=0.2,
                                                                                                    shuffle=False, 
                                                                                                    random_state=0)

    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
            scribble_path = self._scribbles[random.randint(0,950)] # pick randomly from first 1000 scribbles
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
            scribble_path = self._scribbles[idx]
        
        # Read image, mask and scribble
        image = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask[mask<=127] = 0
        mask[mask>127] = 1
        mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=0)
        scribble = Image.open(scribble_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((224, 224)), 
                                               transforms.CenterCrop((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        transforms_mask = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.CenterCrop((224,224)),
                                              transforms.ToTensor()])
        
        # Convert to torch tensors
        image = transforms_image(image)
        mask = torch.from_numpy(mask)
        scribble = transforms_mask(scribble)
        
        ###############################
        # TODO
        #partial_image1 = image * mask * cmask
        #partial_image2 = image * cmask * (1 - mask)
        ###############################
        
        # Masked image
        partial_image1 = image * (torch.max(scribble) - scribble) 
        partial_image2 = image * scribble
        
        sample = {'image': image, 
                  'mask': mask, 
                  'partial_image1': partial_image1,
                  'partial_image2': partial_image2}
        return sample


class GLAS_dataloader(Dataset):
    """
    GLAS data loader with Irregular Masks Dataset (https://arxiv.org/abs/1804.07723)
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        if self.is_train:
            self._input_folder = os.path.join(self._data_folder, "train", 'img')
            self._label_folder = os.path.join(self._data_folder, "train", 'labelcol')
            self.train_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.train_labels = sorted(glob.glob(self._label_folder + "/*.png"))
        else:
            self._input_folder = os.path.join(self._data_folder, "test", 'img')
            self._label_folder = os.path.join(self._data_folder, "test", 'labelcol')
            self.test_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.test_labels = sorted(glob.glob(self._label_folder + "/*.png"))
        
        self._scribbles_folder = os.path.join(self._data_folder, 'SCRIBBLES')
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))[:1000] # For heavy masking [::-1]

    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
            scribble_path = self._scribbles[random.randint(0,950)] # pick randomly from first 1000 scribbles
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
            scribble_path = self._scribbles[idx]
            
        # Read image, mask and scribble
        image = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask[mask<=127] = 0
        mask[mask>127] = 1
        mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=0)
        scribble = Image.open(scribble_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((224, 224)), 
                                               transforms.CenterCrop((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        transforms_mask = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.CenterCrop((224,224)),
                                              transforms.ToTensor()])
        
        # Convert to torch tensors
        image = transforms_image(image)
        mask = torch.from_numpy(mask)
        scribble = transforms_mask(scribble)

        # Masked image
        partial_image1 = image * (torch.max(scribble) - scribble) 
        partial_image2 = image * scribble
        
        sample = {'image': image, 
                  'mask': mask, 
                  'partial_image1': partial_image1,
                  'partial_image2': partial_image2}
        return sample

    
class CVCLINICDB_dataloader(Dataset):
    """
    CVCClinicDB data loader with Irregular Masks Dataset (https://arxiv.org/abs/1804.07723)
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        self._input_folder = os.path.join(self._data_folder, 'Original')
        self._label_folder = os.path.join(self._data_folder, 'GroundTruth')
        self._scribbles_folder = os.path.join(self._data_folder, 'SCRIBBLES')
        self._images = glob.glob(self._input_folder + "/*.png")
        self._labels = glob.glob(self._label_folder + "/*.png")
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))[:1000] # For heavy masking [::-1]

        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(self._images, 
                                                                                                    self._labels,
                                                                                                    test_size=0.1,
                                                                                                    shuffle=False,
                                                                                                    random_state=0)
    
    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
            scribble_path = self._scribbles[random.randint(0,950)] # pick randomly from first 1000 scribbles
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
            scribble_path = self._scribbles[idx]
            
        # Read image, mask and scribble
        image = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask[mask<=127] = 0
        mask[mask>127] = 1
        mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=0)
        scribble = Image.open(scribble_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((224, 224)), 
                                               transforms.CenterCrop((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        transforms_mask = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.CenterCrop((224,224)),
                                              transforms.ToTensor()])
        
        # Conver to torch tensors
        image = transforms_image(image)
        mask = torch.from_numpy(mask)
        scribble = transforms_mask(scribble)

        # Masked image
        partial_image1 = image * (torch.max(scribble) - scribble) 
        partial_image2 = image * scribble
        
        sample = {'image': image, 
                  'mask': mask, 
                  'partial_image1': partial_image1,
                  'partial_image2': partial_image2}
        return sample
    

class RITE_dataloader(Dataset):
    """
    RITE data loader with Irregular Masks Dataset (https://arxiv.org/abs/1804.07723)
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        if self.is_train:
            self._input_folder = os.path.join(self._data_folder, "train", 'img')
            self._label_folder = os.path.join(self._data_folder, "train", 'labelcol')
            self.train_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.train_labels = sorted(glob.glob(self._label_folder + "/*.png"))
        else:
            self._input_folder = os.path.join(self._data_folder, "test", 'img')
            self._label_folder = os.path.join(self._data_folder, "test", 'labelcol')
            self.test_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.test_labels = sorted(glob.glob(self._label_folder + "/*.png"))
        
        self._scribbles_folder = os.path.join(self._data_folder, 'SCRIBBLES')
        self._scribbles = sorted(glob.glob(self._scribbles_folder + "/*.png"))[:1000] # For heavy masking [::-1]

    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
            scribble_path = self._scribbles[random.randint(0,950)] # pick randomly from first 1000 scribbles
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
            scribble_path = self._scribbles[idx]
            
        # Read image, mask and scribble
        image = Image.open(img_path).convert('RGB')
        mask = cv2.imread(mask_path, 0)
        mask[mask<=127] = 0
        mask[mask>127] = 1
        mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=0)
        scribble = Image.open(scribble_path).convert('P')
        
        transforms_image = transforms.Compose([transforms.Resize((224, 224)), 
                                               transforms.CenterCrop((224,224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        transforms_mask = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.CenterCrop((224,224)),
                                              transforms.ToTensor()])
        # Convert to torch tensors
        image = transforms_image(image)
        mask = torch.from_numpy(mask)
        scribble = transforms_mask(scribble)

        # Masked image
        partial_image1 = image * (torch.max(scribble) - scribble) 
        partial_image2 = image * scribble
        
        sample = {'image': image, 
                  'mask': mask, 
                  'partial_image1': partial_image1,
                  'partial_image2': partial_image2}
        return sample
