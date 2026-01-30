import torch
def f(x):
    return x.sin() + x.cos()

print(torch.__version__)
f_c = torch.jit.script(f)