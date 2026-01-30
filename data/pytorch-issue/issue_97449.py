import torch

def fun(x):
    x = x.to("cuda:0")
    
fun_compiled = torch.compile(fun)

x = torch.randn(1)
out = fun_compiled(x)