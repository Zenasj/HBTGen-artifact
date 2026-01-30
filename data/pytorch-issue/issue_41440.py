import torch.nn as nn

import torch
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op('Custom', input, outputs=2)
    @staticmethod
    def forward(ctx, input):
        return input, input
class Custom(torch.nn.Module):
    def forward(self, input):
        return CustomFunction.apply(input)
print ('torch.__version__', torch.__version__)
model = Custom()
batch = torch.FloatTensor(1, 3)
torch.onnx.export(model, batch, "test.onnx", verbose=True)