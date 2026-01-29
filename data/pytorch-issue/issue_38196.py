# torch.rand(B, 1, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3, 5)  # Matches input shape after flattening

    def forward(self, x):
        # Reshape (B, 1, 3) → (B, 3) for linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create valid input by stacking tensors instead of using torch.Tensor(list)
    tensors_list = [torch.rand(1, 3) for _ in range(2)]  # Matches original example's tensor shape
    return torch.stack(tensors_list)  # Produces (2, 1, 3) tensor

