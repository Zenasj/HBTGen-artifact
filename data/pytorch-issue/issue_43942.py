import torch
import torch.nn as nn

class MyModule(torch.nn.Module):
    def forward(self) -> Any:
        if self.training:
            return 'xx'
        else:
            return {}

class MyModule(torch.nn.Module):
    def forward(self) -> Any:
        return 'xx' if self.training else {}