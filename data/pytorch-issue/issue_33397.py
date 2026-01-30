import torch
import torch.nn as nn

class MyModule(torch.nn.Module):

    @ignore
    def python_func(self, x: Tensor) -> Tensor:
        return x + x

    def forward(self, input: Tensor) -> Tensor:
        return self.python_func(input)


m = script(MyModule())
print(m.code)