import torch
import torch.jit
import torch.nn as nn


class CustomModule(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * 3.0


if __name__ == "__main__":
    model = torch.jit.script(CustomModule())
    torch.jit.save(model, "model.pt")

class CustomModule(nn.Module):
    def forward(self, x: torch.Tensor):  # type: ignore
        return x * 3.0

class CustomModule(nn.Module):
    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        x = args[0]
        return x * 3.0