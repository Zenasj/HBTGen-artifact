def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.t()
    sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)

class L1CDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)

        # cdist.forward does not have the memory problem
        return torch.cdist(x1, x2, p=1)

    @staticmethod
    def backward(ctx, grad_dist):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_x1 = grad_x2 = None

        # Retrieve saved values
        x1, x2 = ctx.saved_tensors
        dims = x1.shape[1]

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_x1 = torch.empty_like(x1)
        if ctx.needs_input_grad[1]:
            grad_x2 = torch.empty_like(x2)

        if any(ctx.needs_input_grad):
            for i in range(dims):
                #: sign: shape: (n1, n2)
                sign = torch.sign(x1[:, None, i] - x2[None, :, i])
                if ctx.needs_input_grad[0]:
                    grad_x1[:, i] = torch.sum(grad_dist * sign, dim=1)
                if ctx.needs_input_grad[1]:
                    grad_x2[:, i] = -torch.sum(grad_dist * sign, dim=0)

        return grad_x1, grad_x2

cdist1 = L1CDist.apply

def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.transpose(-1,-2)
    sq_dist = x_sq_norm + y_sq_norm.unsqueeze(dim=-2) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)

import torch


def test(b,n,m):
    '''
    b: batch size
    n: number of samples
    m: number of dimensions
    '''
    torch.cuda.empty_cache()
    x = torch.rand(b,n,m, device='cuda', requires_grad=True)
    y = torch.cdist(x,x)
    torch.cuda.empty_cache()
    print(' forward memory', torch.cuda.memory_cached()/2**20)
    y.sum().backward()
    print('backward memory', torch.cuda.memory_cached()/2**20)
    del x,y