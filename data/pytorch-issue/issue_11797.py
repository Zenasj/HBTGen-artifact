import torch
import scipy.sparse.linalg
from torch.autograd import gradcheck

class MyCustomFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, b):
        ctx.intermediate = (A,b)
        return torch.tensor(A.dot(b.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        # Not implemented
        return None, grad_output

N = 250
A = 2*scipy.eye(N)                  # gradcheck works, but it returns false because the backward is not implemented.
# A = 2*scipy.sparse.eye(N).todia()   # gradcheck gives error "'dia_matrix' object is not subscriptable"
# A = 2*scipy.sparse.eye(N).tocoo()   # gradcheck gives error "'coo_matrix' object is not subscriptable"
# A = 2*scipy.sparse.eye(N).tocsr()   # gradcheck gives error "RecursionError: maximum recursion depth exceeded while calling a Python object".
# A = 2*scipy.sparse.eye(N).tolil()   # gradcheck gives error "RecursionError: maximum recursion depth exceeded in comparison".
x = torch.ones(N, dtype=torch.float64, requires_grad=True)
input = (A,x)
test = gradcheck(MyCustomFunction.apply, input, eps=1e-6, atol=1e-4)
print(test)

def make_jacobian(input, num_out):
    if scipy.sparse.isspmatrix(input):  # Added
        return None                                # Added
    elif isinstance(input, torch.Tensor): # Replaced if by elif
        if not input.is_floating_point():
            return None
        if not input.requires_grad:
            return None
        return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
    elif isinstance(input, Iterable):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None

import scipy.sparse   # Added

def iter_tensors(x, only_requiring_grad=False):
    if scipy.sparse.isspmatrix(x):  # Added
        yield x                                 # Added
    elif isinstance(x, torch.Tensor):   # Replaced if by elif
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result