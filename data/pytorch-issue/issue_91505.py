import torch.nn as nn

import torch

class Model(torch.nn.Module):

    def forward(self, tensor: torch.Tensor):  # (4, 4)
        tensor = tensor.unfold(0, 2, 1)  # (3, 4, 2)
        tensor = tensor.unsqueeze(1)  # (3, 1, 4, 2)
        tensor = tensor.permute([0, 2, 3, -3])  # (3, 4, 2, 1)

        return tensor

compiled_model = torch.compile(Model())
a_in = torch.randn((4, 4))
a_out = compiled_model(a_in)