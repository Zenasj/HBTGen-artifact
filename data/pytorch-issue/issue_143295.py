# torch.rand(10, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Element-wise comparison (angle of entire tensor vs per-element angle)
        angles_full = x.angle()
        angles_individual = torch.stack([x[i].angle() for i in range(len(x))])
        element_mismatch = (angles_full != angles_individual).any()
        
        # Concatenate comparison (concat before vs after angle computation)
        part1 = x[:5]
        part2 = x[5:]
        concat_before = torch.cat([part1, part2], dim=0).angle()
        concat_after = torch.cat([part1.angle(), part2.angle()], dim=0)
        concat_mismatch = (concat_before != concat_after).any()
        
        return torch.tensor([element_mismatch, concat_mismatch], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.complex(
        torch.arange(10, dtype=torch.float32),
        torch.arange(10, dtype=torch.float32),
    )

