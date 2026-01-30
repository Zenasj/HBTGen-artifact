class MaxUnpool2dFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride):
        return g.op('MaxUnpool', input, indices)

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride):
        return F.max_unpool2d(input, indices, kernel_size, stride)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxUnpool2dFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride):
        return g.op('MaxUnpool', input, indices)

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride):
        return F.max_unpool2d(input, indices, kernel_size, stride)
   
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, input, indices):
        return MaxUnpool2dFunc.apply(input, indices, 2, 2)
    

inp = torch.randn([5, 3, 6, 8])
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
output, indices = pool(inp)

model = MyModel()

with torch.no_grad():
    torch.onnx.export(model, 
                      (output, indices), 
                      'model.onnx',
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
                      )