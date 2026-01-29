import torch
import torchvision.transforms as transforms
import random

# torch.rand(1, 3, 100, 100, dtype=torch.float32)
class MyModel(torch.nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.seed = seed
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(30),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        return self.transforms(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 100, 100, dtype=torch.float32)

