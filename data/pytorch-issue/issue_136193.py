import torch
import torch._dynamo.utils
from torch._dynamo import compiled_autograd

def test_compiled_autograd():
        device = "cpu"
        backend = "inductor"

        grads = []
        
        def hook_fn(grad):
            #If the line below is commented out or changed to, for example, grads.append(grad), there is no issue
            grads.append(grad)
            return grad
        
        def fn(x, y):
            x.register_hook(hook_fn)
            return x + y
        
        def compiler_fn(gm):
            return torch.compile(gm, backend=backend)

        torch._dynamo.reset()
    
        with compiled_autograd.enable(compiler_fn):
            x = torch.tensor([0.5, 0.5], device=device, requires_grad=True)
            y = torch.tensor([0.5, 0.5], device=device, requires_grad=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0], device=device)
            out.backward(grad_out)

test_compiled_autograd()