# torch.rand(1, 1, 1, 1, dtype=torch.int32)  # Input shape: B=1, C=1, H=1, W=1 (scalar integer)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.list_container = [1, 2, 3]       # List[int]
        self.tuple_container = (4, 5, 6)      # Tuple[int, int, int]
        self.dict_container = {7: 7, 8: 8}    # Dict[int, int]

    def forward(self, x):
        key = x.item()  # Extract scalar integer from tensor input
        list_result = key in self.list_container
        tuple_result = key in self.tuple_container
        dict_result = key in self.dict_container
        # Return combined results as a boolean tensor
        return torch.tensor([list_result, tuple_result, dict_result], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random integer in [0, 9] as a tensor matching required shape
    return torch.randint(0, 10, (1, 1, 1, 1), dtype=torch.int32)

