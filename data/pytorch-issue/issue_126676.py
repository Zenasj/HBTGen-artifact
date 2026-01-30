import torch

class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = x @ w
        ctx.save_for_backward(x, w)
        return y

    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        print(hasattr(x, "aaa"))  # the custom attribute is deleted
        return grad_output @ w.t(), x.t() @ grad_output

def pack_hook(tensor):
    tensor.aaa = 1
    return tensor
        
def unpack_hook(tensor):
    print(tensor.aaa)  # the custom attribute still exists before unpack returns
    return tensor

x = torch.randn(100, 200).bfloat16().cuda().requires_grad_()
w = torch.randn(200, 300).bfloat16().cuda().requires_grad_()
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = f.apply(x, w).sum()
    y.backward()