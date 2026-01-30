import torch

print(torch.nonzero(a, as_tuple=False)) #ok
print(torch.nonzero(a, out=out)) #warns
print(torch.nonzero(a, as_tuple=False, out=out)) #errors out with `TypeError: nonzero() received an invalid combination of arguments - got unrecognized keyword arguments: out`