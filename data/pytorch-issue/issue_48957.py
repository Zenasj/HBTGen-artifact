import torch
import torch.nn as nn

# torch.rand(B, 2, dtype=torch.float32)  # Input shape inferred from Dataset's __getitem__
VAR = 0  # Global variable as in data.py

def set_var():
    global VAR
    VAR = 1

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy model to satisfy structure requirements
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    return torch.rand(2, 2, dtype=torch.float32)  # Example batch size 2

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = torch.tensor([[0, 1], [2, 3]])
        self.y_data = torch.tensor([4, 5])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        X = self.x_data[index]
        y = self.y_data[index]
        print(f"... In MyDataset.__getitem__, VAR is currently {VAR} (should be 1)")
        return X, y

