ceil, floor, log, log10, log2, sqrt, rsqrt, trunc

import torch
 
ops_list = ['angle', 'sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'asin', 'acos', 'atan', 'ceil', 'floor', 'exp', 'expm1', 'log', 'log10', 'log1p', 'log2', 'sqrt', 'rsqrt', 'trunc', 'erf', 'erfc', 'erfinv', 'lgamma', 'digamma', 'sigmoid']

t = torch.tensor((1,), device='cuda', dtype=torch.complex128)
out_t = torch.tensor((1,), device='cuda', dtype=torch.complex64)
 
for op in ops_list:
    try:
        print(getattr(torch, op)(t, out=out_t), op)
    except RuntimeError as e:
        print(op, e)