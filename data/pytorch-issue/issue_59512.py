# torch.rand(3, 1, dtype=torch.float32)
import torch
from torch import nn
from torch.utils.data import TensorDataset, Subset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # Dummy layer for demonstration

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    dataset = TensorDataset(torch.tensor([1, 2, 3], dtype=torch.float32))
    subset1 = Subset(dataset, list(range(3)))
    subset2 = Subset(subset1, list(range(3)))
    indices = subset2.indices
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    data = dataset[indices_tensor]  # Access via tensor indices to avoid slicing error
    # Extract tensor from dataset and reshape for model input
    tensor_data = torch.stack([item[0] for item in data]).view(-1, 1)
    return tensor_data

