# torch.rand(512, 512, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class FakeParam(object):
    def __init__(self, data: torch.Tensor):
        self.data = data

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.param = FakeParam(
            torch.empty(
                output_size,
                input_size,
                device=torch.device("cuda"),
                dtype=torch.float32,
            ),
        )

    def forward(self, input_):
        # Reshape to 2D (batch_size, input_size) for matmul
        input_ = input_.view(input_.shape[0], -1)
        return torch.matmul(input_, self.param.data)

def my_model_function():
    # Initialize model with input/output size 512 and move to CUDA
    return MyModel(512, 512).cuda()

def GetInput():
    # Generate 4D input matching the expected reshaped 2D form
    return torch.rand(512, 512, 1, 1, dtype=torch.float32, device="cuda")

