import torch.nn as nn

py
import torch

Gradients = torch.Tensor | tuple[torch.Tensor, ...]

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.some_param = torch.nn.Parameter(torch.tensor(1.0))
        self.register_full_backward_hook(self.full_backward_hook)

    @staticmethod
    def full_backward_hook(
        module: torch.nn.Module,
        grad_in: Gradients,
        grad_out: Gradients,
    ) -> None:
        print("full_backward_hook called with", module, grad_in, grad_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.some_param

model = MyModule()

output = model.forward(torch.tensor(5))
loss = output.sum()

print("model.some_param.grad:", model.some_param.grad)

print("calling backward")
loss.backward()

print("model.some_param.grad:", model.some_param.grad)