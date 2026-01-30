import torch.nn as nn

import torch

class CopyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_features, weight):
        ctx.output_features=input_features.clone()
        return ctx.output_features
    @staticmethod
    def backward(ctx, grad_output):
        print('backward is never called; that is OK.')

for _ in range(1000000):
    weight = torch.nn.Parameter(torch.Tensor(100,100).normal_())
    input=torch.FloatTensor(100,100).uniform_().cuda()
    output = CopyFunction.apply(input,weight)
    print(output.shape, output.type())