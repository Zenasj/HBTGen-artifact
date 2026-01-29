# torch.randint(0, 50000, (B,), dtype=torch.int64)
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

class MyModel(torch.nn.Module):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def forward(self, indices):
        results = []
        for idx in indices:
            idx = idx.item()
            raw_data = self.dataset.data[idx]
            img = Image.fromarray(raw_data)
            transformed_data = self.transform(img)
            data_via_getitem = self.dataset.__getitem__(idx)[0]
            results.append(torch.allclose(transformed_data, data_via_getitem, atol=1e-5))
        return torch.tensor(results, dtype=torch.bool)

def my_model_function():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    return MyModel(dataset, train_transforms)

def GetInput():
    return torch.randint(0, 50000, (1,), dtype=torch.int64)

