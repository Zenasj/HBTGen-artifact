import torch

import torch.nn as nn


class ExampleIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('Indexing:', type(x.shape[-1]))
        print('Using size:', type(x.size(-1)))
        return x


x = torch.rand((1, 3, 224, 244))
ei = ExampleIdentity()

print('\n\nRunning normally:')
ei(x)

print('\n\nVia onnx.export:')
with open('/tmp/output.onnx', 'wb') as outf:
    torch.onnx.export(ei, x, outf)