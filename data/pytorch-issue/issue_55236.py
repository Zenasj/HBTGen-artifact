import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, LSUN
from torchvision.transforms import transforms

path = './data/'
dset = MNIST(root=path,
                train=True,
                transform=transforms.ToTensor(),
                download=True)

d_loader = DataLoader(dataset=dset,
                        batch_size=args.batch_size,
                        shuffle=True)