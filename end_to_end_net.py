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
import argparse

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
            split (string): Split the dataset into training or testing set
        """
        self.csv_file = csv_file
        self.handover = pd.read_csv(self.csv_file)
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

"""
TRAINING LOOP
"""
def do_train(model, device, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a dict of [inputs, class]
            inputs, cls = data['image'].to(device), data['class'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
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
def do_test(model, device, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, cls = data['image'].to(device), data['class'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (len(testloader),
        100 * correct / total))

"""
MAIN
"""
def main(args):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomResizedCrop((720, 1280), scale=(0.7,1.0)),
                                    transforms.ToTensor()])

    handover_train = HandoverDataset(csv_file=args.train_csv[0],
                                        img_dir=args.img_folder[0],
                                        transform=transform,
                                    )

    transform = transforms.Compose([transforms.ToTensor()])

    handover_test = HandoverDataset(csv_file=args.test_csv[0],
                                        img_dir=args.img_folder[0],
                                        transform=transform,
                                    )

    trainloader = DataLoader(handover_train, batch_size=2, shuffle=True, num_workers=2)
    testloader = DataLoader(handover_test, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50()
    resnet50 = resnet50.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)
    epochs = 10

    do_train(resnet50, device, trainloader, criterion, optimizer, epochs)
    do_test(resnet50, device, testloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train the end-to-end network"
    )
    parser.add_argument(
        "--img_folder",
        default="./",
        nargs=1,
        metavar="IMAGE_FOLDER",
        help="Path to image folder",
        type=str
    )
    parser.add_argument(
        "--train_csv",
        nargs=1,
        metavar="TRAIN_CSV",
        help="Path to train set csv",
        type=str
    )
    parser.add_argument(
        "--test_csv",
        nargs=1,
        metavar="TEST_CSV",
        help="Path to test set csv",
        type=str
    )

    args = parser.parse_args()
    # print(args.train_csv)

    main(args)
