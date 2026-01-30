import torch
f = torch.special.polygamma
cf = torch.compile(f)
n=1
input=torch.tensor([-1.0])

eager = f(n,input)
compile = cf(n,input)
expected = f(n, input.to(torch.float64))
print(f"Eager: {eager}, dtype: {eager.dtype}")
print(f"Compile: {compile}, dtype: {compile.dtype}")
print(f"Expected: {expected}, dtype: {expected.dtype}")