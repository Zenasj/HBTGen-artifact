import torch.nn as nn

from typing import Any
import torch

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> Any:
        if input.shape[0] == 1:
            return 1
        else:
            return "3"

if __name__ == "__main__":
    m = TestModule()
    m_scripted = torch.jit.script(m)
    print(m(torch.randn(1, 2)))
    print(m_scripted(torch.randn(1, 2)))  # segfault!
    print(m_scripted(torch.randn(2, 2)))  # segfault!