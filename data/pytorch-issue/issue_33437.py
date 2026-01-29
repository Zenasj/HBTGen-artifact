import torch
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encapsulate both models as submodules
        self.demo1 = nn.Sequential(
            nn.Conv2d(3, 20, 3),
            nn.Conv2d(3, 20, 3)
        )
        self.demo2 = nn.Sequential(
            nn.Conv2d(3, 20, 3),
            nn.Conv2d(3, 20, 3)
        )
        # Note: Original Demo1/Demo2 had separate conv layers but identical structure
    
    def forward(self, x):
        # Run both models and compare outputs
        list_out = [self.demo1[0](x), self.demo2[0](x)]  # Mimic list output structure
        dict_out = {'out1': self.demo1[1](x), 'out2': self.demo2[1](x)}  # Mimic dict output structure
        # Perform comparison using allclose (structure comparison logic)
        is_same = torch.allclose(list_out[0], dict_out['out1']) and torch.allclose(list_out[1], dict_out['out2'])
        return torch.tensor(is_same, dtype=torch.bool)  # Return comparison result

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 512, 512, dtype=torch.float32)

