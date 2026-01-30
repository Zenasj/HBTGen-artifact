import torch
import torch.nn as nn

class Interpreter(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        
    # ... Hooks to capture calls etc ...
    
    
def custom_compile(gm : torch.fx.GraphModule, _):
    def wrapper(*args, **kwargs):
        return Interpreter(gm).run(*args, **kwargs)

    return make_boxed_func(wrapper)

class Wat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float()

x = torch.randn((8, 128), dtype=torch.float16, device='cuda')
print('Input shape:', x.shape)

wat = Wat()
compiled_wat = torch.compile(wat, backend=custom_compile)

y = compiled_wat(x)
print('Wat:', y.shape)

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

x = torch.randn((8, 128), dtype=torch.float16, device='cuda')
print('Input shape:', x.shape)

ident = Identity()
compiled_ident = torch.compile(ident, backend=custom_compile)

y = compiled_ident(x)
print('Ident:', y.shape)

def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True
    return g

def make_boxed_func(f):
    def g(*args):
        return f(*args)

    g._boxed_call = True
    return g