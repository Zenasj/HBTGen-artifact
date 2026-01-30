py
import torch
v = torch.tensor([0., 0., 0.], requires_grad=True)

@torch.compile(backend="aot_eager", fullgraph=True)
def f(v):
    lr = 0.01
    #    simulate a simple SGD update
    h = v.register_hook(lambda p: p.add_(p.grad, alpha=-lr))
    
    return v.clone(), h

print(v)
k, h = f(v)
v.backward(torch.tensor([1., 2., 3.]))
print(v)