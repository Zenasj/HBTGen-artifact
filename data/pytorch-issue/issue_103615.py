py
import torch
import torch._dynamo

@torch.compile(backend='aot_eager', fullgraph=True)
def f(x):
    @torch._dynamo.allow_in_graph
    def apply_func(x):
        return x.clone()
    
    return apply_func(x)

x = torch.randn([], requires_grad=True)

f(x)