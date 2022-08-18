import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import re
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


root = './weizmann_horse_db'
class HorseDataset(Dataset):
    def __init__(self, root: str, idx, transforms=None):
        super(HorseDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.idx = idx  

        self.imgs = np.array(os.listdir(os.path.join(root, "horse")))[idx]
        self.masks = np.array(os.listdir(os.path.join(root, "mask")))[idx]

    def __getitem__(self, idx):
        img_path = self.root + '/' + 'horse' + '/' + self.imgs[idx]
        mask_path = self.root + '/' + 'mask' + '/' + self.masks[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)

