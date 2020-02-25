import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = self.sigmoid(x)
        return output

"""
TRAINING LOOP
"""
def do_train(model, device, trainloader, criterion, optimizer, epochs, weights_pth):
    model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a dict of [inputs, class]
            inputs, cls = data['image'].to(device), data['class'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, cls.view(-1,1).float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %3d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    torch.save(model.state_dict(), weights_pth)

    print('Finished Training')

"""
TESTING LOOP
"""
def do_test(model, device, testloader, weights_pth):
    model.load_state_dict(torch.load(weights_pth))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, cls = data['image'].to(device), data['class'].to(device)
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (len(testloader),
        100 * correct / total))

def main(args):
    input_size = 29
    output_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size, output_size=output_size)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)
    epochs = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--json-path",
        default="./",
        nargs=1,
        metavar="JSON_PATH",
        help="Path to the json file",
        type=str
    )
