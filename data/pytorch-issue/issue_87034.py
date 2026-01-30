import torch
import torch.nn as nn

class MyRaggedModule(nn.Module):
        def __init__(self, sizes, device=None):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.nested.nested_tensor([
                torch.empty(size, device=device, dtype=torch.float)
                for size in sizes
            ]))
            self.reset_parameters()

        def reset_parameters(self):
            with torch.no_grad():
                bits = self.weights.unbind()
                for bit in bits:
                    torch.nn.init.normal_(bit)