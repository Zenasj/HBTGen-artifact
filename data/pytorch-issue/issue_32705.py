# torch.rand(3, 8, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.register_buffer('indices', torch.zeros((2, 3), dtype=torch.long))
        
    def forward(self, x):
        # Create sparse tensor (note: indices may be invalid for 3D tensors)
        comb = torch.sparse.FloatTensor(self.indices, x, (4, 4, 8)).to_dense()
        big = F.pad(comb, (0, 0, 1, 1, 1, 1))  # Pad last 3 dimensions
        shaped = big.view(-1, 8).permute(1, 0).unsqueeze(0)
        res = F.fold(shaped, output_size=(5,5),
                    kernel_size=(2, 2), padding=(1, 1))
        return res

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 8, dtype=torch.float, requires_grad=True)

