# torch.rand(1, dtype=torch.float32)  # Inferred input shape: (batch_size=1,)

import torch
import logging

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)
        self.val = 0  # If this is 0, we trigger recompilation. If this is torch.tensor([1]), we do not.

    def pre_forward(self, module, args, kwargs):
        if torch._utils.is_compiling():
            self.val += 1
            return args, kwargs

        logging.warning("path B")
        return args, kwargs

    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([1], dtype=torch.float32)

