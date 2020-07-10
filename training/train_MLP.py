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
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.autograd import Variable
import json
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class HandoverDataset(Dataset):
    """Handover dataset"""

    def __init__(self, json_file):
        self.json_file = json_file

        with open(json_file) as f:
            json_file = json.load(f)
            # print(json_file)
            self.annos = json_file

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.annos[idx]["object_detection"]["pred_boxes"]:
            obj_det = self.annos[idx]["object_detection"]["pred_boxes"]
            obj_det = np.asarray(obj_det)
            obj_det = obj_det.flatten()
        else:
            obj_det = [-999,-999,-999,-999]
            obj_det = np.asarray(obj_det)

        key_det = self.annos[idx]["keypoint_detection"]["pred_keypoints"]
        key_det = np.asarray(key_det)
        key_det = key_det[0:11, 0:2]
        key_det = key_det.flatten()

        if self.annos[idx]["head_pose_estimation"]["predictions"]:
            hp_est = self.annos[idx]["head_pose_estimation"]["predictions"]
            hp_est = np.asarray(hp_est)
            hp_est = hp_est.flatten()
            hp_est = hp_est[0:3]
        else:
            hp_est = np.asarray([-999, -999, -999])

        label = self.annos[idx]["label"]

        anno_list = np.concatenate((obj_det, key_det, hp_est))
        sample = {"annotations": anno_list, "class": label}

        return sample

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024,2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
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
            inputs, cls = data['annotations'].to(device), data['class'].to(device)
            inputs = Variable(inputs).float().cuda()
            cls = Variable(cls).float().cuda()
            # print(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, cls.view(-1,1))
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
            inputs, cls = data['annotations'].to(device), data['class'].to(device)
            inputs = Variable(inputs).float().cuda()
            cls = Variable(cls).float().cuda()
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += cls.size(0)
            correct += (predicted.flatten() == cls).sum().item()

            # print(outputs, predicted, cls)

    print('Accuracy of the network on the %d test images: %5f %%' % (len(testloader),
        100 * correct / total))

def main(args):
    input_size = 29
    output_size = 1

    handover_dataset = HandoverDataset(args.json_path[0])
    handover_train, handover_test = random_split(handover_dataset, (round(0.8*len(handover_dataset)), round(0.2*len(handover_dataset))))

    trainloader = DataLoader(handover_train, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(handover_test, batch_size=1, shuffle=False, num_workers=2)

    weights_pth = args.weights_path[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size, output_size=output_size)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    epochs = 500

    if args.eval_only:
        do_test(model, device, testloader, weights_pth)
    else:
        do_train(model, device, trainloader, criterion, optimizer, epochs, weights_pth)
        do_test(model, device, testloader, weights_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--json-path",
        default="./",
        nargs="+",
        metavar="JSON_PATH",
        help="Path to the json file",
        type=str
    )
    parser.add_argument(
        "--weights-path",
        default="./",
        nargs="+",
        metavar="WEIGHTS_PATH",
        help="Path to the weights file",
        type=str
    )
    parser.add_argument(
        "--eval-only",
        help="set model to evaluate only",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
