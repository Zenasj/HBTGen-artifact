import torch.nn as nn

import torchvision.transforms.v2 as transforms
# train_transforms = transforms.Compose( 
train_transforms = torch.nn.Sequential(  
    # [
        transforms.RandomResizedCrop((448, 448)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
    # ]
)
train_transforms = torch.compile(train_transforms, disable=False)