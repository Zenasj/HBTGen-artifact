from pathlib import Path
from typing import List

import torch
print(torch.__version__)

x1 = (torch.randn(1, 10, 60, 80), )
x2 = ([torch.randn(1, 10, 60, 80), torch.randn(1, 10, 30, 40), torch.randn(1, 10, 15, 20)], )

def test1(outs: torch.Tensor) -> torch.Tensor:
    return outs.flatten(start_dim=2)

def test2(outs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.flatten(start_dim=2) for x in outs]

test1 = torch.jit.script(test1)
test1(*x1)
print(type(test1))
torch.onnx.export(test1, x1, Path('sth.onnx'),
                  opset_version=15)

test2 = torch.jit.script(test2)
test2(*x2)
print(type(test2))
torch.onnx.export(test2, x2, Path('sth.onnx'),
                  opset_version=15)

assert x.shape==(B*C, 4, H, W) # for example
y=x[:, 0, :, :] # ONNX will know the shape (and rank) and allow this

x=x.reshape(B*C, 4, H, W)
y=x[:, 0, :, :] # ONNX will know the shape (and rank) and allow this