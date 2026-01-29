# torch.rand(B=1, C=2, H=1, W=4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.index0 = 0
        self.index1 = 1
        self.index2 = 2
        self.index3 = 3

    def forward(self, x):
        arg1_1 = x[0, 0, 0, :]  # Extract arg1_1 from input tensor
        arg2_1 = x[0, 1, 0, :]  # Extract arg2_1 from input tensor

        # Compute conditions for masked_subblock1
        ge = (self.index1 >= 1)
        lt = (self.index1 < 3)
        and_ = ge and lt  # True when index1 is 1 or 2 (since index1=1)

        # Compute masked_subblock2 value (arg1_1 at index2)
        masked_subblock2_val = arg1_1[self.index2]

        # Load from arg2_1 at index3 and apply dtype conversion (fix applied)
        load_val = arg2_1[self.index3]
        to_dtype = load_val.to(torch.bfloat16).to(torch.float32)  # Fix: ensure float32

        # masked_subblock1 result
        masked_subblock1_result = torch.where(
            torch.tensor(and_, dtype=torch.bool),
            masked_subblock2_val,
            to_dtype
        )

        # Body's final where operation (always selects masked_subblock1_result)
        final_result = torch.where(
            torch.tensor(True, dtype=torch.bool),  # index0 <4 is always true
            masked_subblock1_result,
            arg2_1[self.index3].to(torch.bfloat16).to(torch.float32)
        )

        return final_result

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B, C, H, W) matching the model's requirements
    return torch.rand(1, 2, 1, 4, dtype=torch.float32)

