import torch
import torch.nn as nn

x = torch.rand([2, 3, 60, 80, 6]).to('cuda:0')
ag = torch.tensor([[[[[[ 10.,  13.]]],
  [[[ 16.,  30.]]],
  [[[ 33.,  23.]]]]],
[[[[[ 30.,  61.]]],
  [[[ 62.,  45.]]],
  [[[ 59., 119.]]]]],
[[[[[116.,  90.]]],
  [[[156., 198.]]],
  [[[373., 326.]]]]]])

class Test(nn.Module):

    def __init__(self, anchor):
        super().__init__()
        self.anchor=anchor.to('cuda:0')

    def forward(self, x):
        return x[...,2:4]*self.anchor[0]

model=Test(ag).to('cuda:0').eval()
x = x.to('cuda:0')
result = model(x)

fname = 'test.onnx'
torch.onnx.export(model, x, fname, verbose=False, opset_version=12)