import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
from tqdm.auto import tqdm


dataset = FakeData(size=100, transform=transforms.ToTensor(), num_classes=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
epochs = 90
for _ in tqdm(range(epochs)):
    for _ in tqdm(dataloader, leave=False):
        continue