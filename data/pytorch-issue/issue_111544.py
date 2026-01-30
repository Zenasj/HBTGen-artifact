import torch

param = torch.zeros(5)
param2 = torch.zeros(5)

tensor_list = set()
tensor_list.add(param2)
print(param2 in tensor_list)  # False

def fn(param, param2):
    tensor_list = set([param2])
    return param in tensor_list 

ret = torch.compile(fn, onegraph=True)(param, param2)

param = torch.zeros(5)
param2 = torch.zeros(5)

tensor_list = set()
tensor_list.add(param2)
print(param2 in tensor_list)  # False

def fn(param, param2):
    tensor_list = set([param2])
    return param in tensor_list

ret = torch.compile(fn, fullgraph=True)(param, param2)
assert ret == fn(param, param2)   # RuntimeError: Boolean value of Tensor with more than one value is ambiguous