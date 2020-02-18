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
import torch.nn as nn
import torch.optim as optim

EPOCHS = 10

"""
DATALOADER
"""
class HandoverDataset(Dataset):
    """Handover dataset"""

    def __init__(self, csv_file, img_dir, transform=None, split='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): Split the dataset into training or testing set
        """

        if split == 'train':
            self.handover = pd.read_csv('datasets/Handover/classes_train.csv')
        elif split == 'test':
            self.handover = pd.read_csv('datasets/Handover/classes_test.csv')
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
            sample["image"] = self.transform(sample["image"])

        return sample

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop((720, 1280), scale=(0.7,1.0)),
                                transforms.ToTensor()])

handover_train = HandoverDataset(csv_file='datasets/Handover/classes_train.csv',
                                    img_dir='datasets/Handover/handover',
                                    transform=transform,
                                    split='train')

# for i in range(len(handover_train)):
#     sample = handover_train[i]
#
#     print(i, sample['image'].size(), sample['class'])
#
#     if i == 3:
#         break

transform = transforms.Compose([transforms.ToTensor()])

handover_test = HandoverDataset(csv_file='datasets/Handover/classes_test.csv',
                                    img_dir='datasets/Handover/handover',
                                    transform=transform,
                                    split='test')

trainloader = DataLoader(handover_train, batch_size=2, shuffle=True, num_workers=2)
testloader = DataLoader(handover_test, batch_size=1, shuffle=False, num_workers=2)

"""
MODEL
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

resnet50 = models.resnet50()
resnet50 = resnet50.to(device)
# print(resnet50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)

"""
TRAINING LOOP
"""
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a dict of [inputs, class]
        inputs, cls = data['image'].to(device), data['class'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet50(inputs)
        loss = criterion(outputs, cls)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %3d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

"""
TESTING LOOP
"""
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, cls = data['image'].to(device), data['class'].to(device)
        outputs = resnet50(images)
        _, predicted = torch.max(outputs.data, 1)
        total += cls.size(0)
        correct += (predicted == cls).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (len(testloader),
    100 * correct / total))
