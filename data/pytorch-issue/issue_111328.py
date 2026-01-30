import torch.nn as nn

py3
import torch
print(torch.__version__) # Tried with 2.1.0+cu121 (latest) and 2.0.1 (default in google colab). All fails

class Linear(nn.Linear):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, *args, **kwargs):
    return super().forward(*args, **kwargs)

m = Linear(4, 8)
m(torch.randn(3, 4)) # passes

compiled_m = torch.compile(m) # passes
compiled_m(torch.randn(3, 4)) # fails