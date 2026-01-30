torchvision.datasets.cifar.CIFAR10

torchvision.datasets.cifar.CIFAR10.data

torchvision.datasets.cifar.CIFAR10.target

__getitem__()

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


train_dataset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
with_data=train_dataset.data[0,0,0,:]
with_idx,_=train_dataset[0][:,0,0].numpy()

print(with_data == with_idx)

[tasklist]
### Tasks