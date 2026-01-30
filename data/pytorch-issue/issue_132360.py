import torch
import torch.nn as nn

def test_select(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.select(x, x.dim()-1, 0)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)