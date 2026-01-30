import torch
import torch.nn as nn

class MyModule(torch.nn.Module):
    def forward(self, x):
        x = x[:, :, [0, 1]]
        return x
# will get torch.jit.frontend.NotSupportedError: slicing multiple dimensions with sequences not supported yet:

class MyModule(torch.nn.Module):
    def forward(self, x):
        x = x[:, :].index_select(2, torch.tensor([0, 1]))
        # or
        idx = torch.tensor([0, 1])
        x = x[:, :, idx]
        return x