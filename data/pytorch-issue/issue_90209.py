py
import torch

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x.relu_()
        ctx.mark_dirty(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        pass

z = torch.tensor(1., requires_grad=True)
x = z.clone()
y = MyRelu.apply(x)
y is x