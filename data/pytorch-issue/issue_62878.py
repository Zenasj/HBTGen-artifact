import torchvision

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision import models
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import random
import os
import tqdm
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

data_transform = transforms.Compose([transforms.ToTensor()])
train_data_root = r'D:\DeepLearning\Data\train'
train_dataset = ImageFolder(train_data_root, transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                           shuffle=True, num_workers=2)
test_data_root = r'D:\DeepLearning\Data\train'
test_dataset = ImageFolder(test_data_root, transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)
def train(epoch):
    for iteration, (images, labels) in enumerate(train_loader):  ###here is wrong location
        images = images.to(device)
        labels = labels.to(device)
if __name__ == '__main__':
    best_acc = 0
    epoches =12
    for epoch in range(1, epoches + 1):
        train(epoch)
        print("===> Epoch({}/{})".format(epoch,epoches))

for d in train_dataset:
    print(d)