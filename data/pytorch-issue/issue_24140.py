import torch
def f(xsp):
    return torch.sparse.sum(xsp)                                                                                                                                                                                                            
f_jit = torch.jit.script(f)