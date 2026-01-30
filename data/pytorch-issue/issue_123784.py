import torch

def fn(x):
    return x.sin().cos()

fn_c = torch.compile(fn)

x = torch.rand((4, 4))
with torch.inference_mode():
    fn_c(x)

fn_c(x)