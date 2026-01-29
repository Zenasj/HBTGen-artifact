# torch.rand(B, C, H, W, D, dtype=...)  # Inferred input shape: (2, 3, 60, 80, 6)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, anchor):
        super().__init__()
        self.register_buffer('anchor', anchor.to('cuda:0'))

    def forward(self, x):
        return x[..., 2:4] * self.anchor[0]

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    ag = torch.tensor([[[[[[10., 13.]]],
                          [[[16., 30.]]],
                          [[[33., 23.]]]]],
                        [[[[[30., 61.]]],
                          [[[62., 45.]]],
                          [[[59., 119.]]]]],
                        [[[[[116., 90.]]],
                          [[[156., 198.]]],
                          [[[373., 326.]]]]]])
    return MyModel(ag)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand([2, 3, 60, 80, 6], dtype=torch.float32).to('cuda:0')

