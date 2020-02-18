import numpy as np
import os
from skimage import io, transform
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import random

"""
DATALOADER
"""
class HandoverDataset(Dataset):
    """Handover dataset"""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.handover = pd.read_csv('datasets/Handover/classes.csv')
        self.handover_dict = self.handover.to_dict("records")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.handover)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.handover_dict[idx]["file_name"])
        cls = self.handover_dict[idx]["class"]
        image = io.imread(img_name)
        sample = {"image": image, "class":cls}

        if self.transform:
            sample = self.transform(sample)

        return sample


handover_dataset = HandoverDataset(csv_file='datasets/Handover/classes.csv',
                                    img_dir='datasets/Handover/handover')

# fig = plt.figure()
#
# for i in random.sample(range(0,len(handover_dataset)), 4):
#     sample = handover_dataset[i]
#
#     print(i, type(sample['image']), sample['class'])
#
#     plt.imshow(sample['image'])
#     plt.show()
#     plt.figure()

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(handover_dataset[0]['image'].shape[0:1], scale=(0.7,1.0)),
                                transforms.ToTensor()])

"""
MODEL
"""
resnet50 = models.resnet50()
