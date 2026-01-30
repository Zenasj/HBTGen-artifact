import torchvision

from torchvision.datasets import MNIST
from torchvision import transforms

train_dataset = MNIST('data/', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))

import torch
print(torch.__version__)
print(torchvision.__version__)