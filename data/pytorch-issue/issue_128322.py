import torch

def addmv_slice(input, mat, vec, slice_op):
    vec = vec[slice_op]
    res = torch.addmv(input, mat, vec) # traced line: 25
    return res

torch._dynamo.reset()
model_opt = torch.compile(addmv_slice)

input = torch.empty(size=[11]).uniform_(-1, 1)
mat = torch.empty([11, 128]).uniform_(-10.0, 20.0)

vec = torch.empty([384]).uniform_(-10.0, 20.0)
slice_op = slice(None, None, 3)
out = model_opt(input, mat, vec, slice_op)


vec = torch.empty([256]).uniform_(-10.0, 20.0)
slice_op = slice(None, None, 2)
out = model_opt(input, mat, vec, slice_op)