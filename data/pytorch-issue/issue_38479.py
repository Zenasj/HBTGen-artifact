import torch.nn as nn
import torch.nn.functional as F

import torch

class Wrapper():
    def __init__(self,data):
        self.data = data
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        #unwrap inputs if necessary
        def unwrap(v):
            return v.data if isinstance(v,Wrapper) else v
        args = map(unwrap,args)
        kwargs = {k:unwrap(v) for k,v in kwargs.items()}

        return func(*args, **kwargs)



# fixed einsum implementation
from torch import Tensor,_VF
from torch._overrides import has_torch_function,handle_torch_function
def fixed_einsum(equation,*operands):
    if not torch.jit.is_scripting():
        if any(type(t) is not Tensor for t in operands) and has_torch_function(operands):
            # equation is not passed
            # return handle_torch_function(einsum, operands, *operands)
            return handle_torch_function(fixed_einsum, operands, equation,*operands)
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        operands = operands[0]

        # recurse incase operands contains value that has torch function
        #in the original implementation this line is omitted
        return fixed_einsum(equation,*operands)

    return _VF.einsum(equation, operands)


if __name__ == "__main__":
    print(torch.__version__)
    # uncomment to use fixed einsum
    # torch.einsum = fixed_einsum

    #operands are wrapped
    x = Wrapper(torch.randn(5))
    y = Wrapper(torch.randn(4))
    assert torch.allclose(torch.einsum('i,j->ij',x, y),torch.ger(x,y))  # outer product
    print("works with wrapped inputs")    

    #old interface operands is a list
    a = Wrapper(torch.randn(2,3))
    b = Wrapper(torch.randn(5,3,7))
    c = Wrapper(torch.randn(2,7))
    assert torch.allclose(torch.einsum('ik,jkl,il->ij', [a, b, c]),torch.nn.functional.bilinear(a,c,b)) # bilinear interpolation
    print("works with old API operands is list")
    
    #equation is wrapped
    As = Wrapper(torch.randn(3,2,5))
    Bs = Wrapper(torch.randn(3,5,4))
    equation = Wrapper('bij,bjk->bik')
    assert torch.allclose(torch.einsum(equation, As, Bs),torch.matmul(As,Bs)) # batch matrix multiplication
    print("works with equation wrapped")

    #see that it also works with plain tensors
    x = torch.randn(5)
    y = torch.randn(4)
    assert torch.allclose(torch.einsum('i,j->ij',x, y),torch.ger(x,y)) 
    print("works with no wrapped values")