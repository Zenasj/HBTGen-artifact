import torchvision

import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST

mnist = MNIST(root="../data")
train, test = random_split(mnist, [.7, .3])
print(f"full MNIST shape: {mnist.data.shape}")
print(f"train shape: {train.dataset.data.shape}")
print(f"test shape: {test.dataset.data.shape}")

print(f"train shape: {train.dataset.tensors[0].shape}")
print(f"test shape: {test.dataset.tensors[0].shape}")