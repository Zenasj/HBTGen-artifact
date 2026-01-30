import torch

class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1):
        # make sure we don't share storage with our input
        # (just to be clear that that's not what's going wrong)
        tensor2 = tensor1.clone()
        # tensor3 shares storage with tensor2
        # tensor3 = tensor2.view(-1) etc. would also demonstrate the problem
        tensor3 = tensor2.t()
        ctx.save_for_backward(tensor3)
        return tensor3
    @staticmethod
    def backward(ctx, grad):
        # trigger correctness check
        _ = ctx.saved_tensors
        return grad.t().clone()


x = torch.rand(2, 3, requires_grad=True)
y = f.apply(x)
y += 1
# Does not raise an error; this is the bug.
y.backward(torch.rand_like(y))


# In constrast, the following code correctly raises an error.
class ff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1):
        tensor2 = tensor1.clone()
        ctx.save_for_backward(tensor2)
        return tensor2
    @staticmethod
    def backward(ctx, grad):
        _ = ctx.saved_tensors
        return grad.clone()

xx = torch.rand(2, 3, requires_grad=True)
yy = ff.apply(xx)
yy += 1
yy.backward(torch.rand_like(yy))