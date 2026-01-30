import torch
import torchvision

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

ds = MNIST('test',
                   download=True,
                   transform=Compose([
                            ToTensor(),
                            Normalize(
                                (0.1307,), (0.3081,))
                        ]))
dl = DataLoader(ds,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True)
print(next(iter(dl)))
print(next(iter(dl)))