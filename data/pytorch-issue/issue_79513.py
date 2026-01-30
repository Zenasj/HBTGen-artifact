import torch.nn as nn

import torch
import torch.fx

class CustomModule(torch.nn.Module):
    def forward(self, x):
        batch_size, num_channels, h, w = x.shape
        b = x / torch.tensor(num_channels)
        return b

if __name__ == "__main__":
    m = CustomModule()
    m_fx = torch.fx.symbolic_trace(m)  # This will trigger error!

import torch
import torch.fx

def foo(x, num_channels):
    b = x / torch.tensor(num_channels)
    return b

torch.fx.wrap("foo")

class CustomModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, h, w = x.shape
        return foo(x, num_channels)

if __name__ == "__main__":
    m = CustomModule()
    m_fx = torch.fx.symbolic_trace(m)  # No error now

import torch
import torch.fx
import functorch

class CustomModule(torch.nn.Module):
    def forward(self, x):
        batch_size, num_channels, h, w = x.shape
        b = x / torch.tensor(num_channels)
        return b

if __name__ == "__main__":
    m = CustomModule()
    # m_fx = torch.fx.symbolic_trace(m)  # This will trigger error!
    x = torch.rand(2, 3, 4, 4)
    m_fx = functorch.make_fx(m)(x) # Work like a charm!