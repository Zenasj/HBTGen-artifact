import torch,PIL
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import cv2
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from collections import Counter
import copy
print(PIL.__version__)
print(torch.__version__)
print(torchvision.__version__)

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
        transforms.Resize((256,256)),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#         transforms.Normalize((0,0,0), (0.1,0.1,0.1))
    ]),
    'val': transforms.Compose([
#         transforms.RandomResizedCrop(224),
        transforms.Resize((256,256)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5), (1,1,1))
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
data_dir = 'data/m_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=0, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    print(inp.shape) # m_data = torch.Size([3, 260, 1034])
    s(inp.numpy())
    # ants = torch.Size([3, 260, 1034])
    inp = inp.numpy().transpose((1, 2, 0))
#     mean = 0#np.array([0.485, 0.456, 0.406])
#     std = 0.1#np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     s(inp)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
a=inputs.numpy()[0,:,:,:].transpose((1, 2, 0))
s(a)
plt.imshow(a), plt.pause(0.001)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

d1 = 'data/hymenoptera_data'
d2 = 'data/m_data'
im = transforms.ToTensor()(Image.open(d2+'/train/benign/0.png')).numpy().transpose((1,2,0))
s(im)
def s(a):
    print(a.shape, a.max(), a.min(), a.mean(), a.std(), Counter(a.ravel()).most_common()[:10])