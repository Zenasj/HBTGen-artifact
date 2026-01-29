# torch.rand(2, 2, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Trigger the split operations with sections=0 to test crash behavior
        # This will raise an error in PyTorch >=1.11 but crash in older versions
        try:
            h_split = torch.hsplit(x, 0)
            v_split = torch.vsplit(x, 0)
            d_split = torch.dsplit(x, 0)
            # Return first element of each split (if successful)
            return h_split[0] + v_split[0] + d_split[0]
        except RuntimeError as e:
            # Handle error for newer versions, return a dummy tensor
            return torch.zeros_like(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, 4, dtype=torch.float32)

