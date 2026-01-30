import torch
import torch.nn as nn

t_1 = torch.rand(3)
t_2 = torch.ones_like(t_1)
t_1.add_(t_2)

t_1 = torch.rand(3)
p_1 = torch.nn.Parameter(t_1)

class Dropout(torch.nn.Module):

    def forward(self, input) -> torch.Tensor: # type: ignore
        return torch.nn.functional.dropout(
            input,
            p=0.5,
            training=self.training
        )