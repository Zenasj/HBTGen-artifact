import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, index, input):
        y_max = input.scatter_reduce(0, index, x, reduce='amax')
        y_sum = input.scatter_reduce(0, index, x, reduce='sum')
        return y_max, y_sum


model = Model()
model.eval()

src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 8.])
torch.onnx.export(model,
                  (src, index, input),
                  'model.onnx',
                  opset_version=16
                  )