import torch.nn as nn

import torch

def traverse_obj(obj, func):
    if isinstance(obj, (tuple, list)):
        a = type(obj)(traverse_obj(o, func) for o in obj)
        print("return tuple tensor: ", a[0].requires_grad)
        return a
    elif isinstance(obj, dict):
        return {name: traverse_obj(o, func) for name, o in obj.items()}
    elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        b = func(obj)
        print("return tensor: ", b.requires_grad)
        return b

def to_mem_format(mem_format, inputs):
    def inner_to_mem_format(obj):
        old_requires_grad = obj.requires_grad
        _tensor = obj.clone().to(memory_format=mem_format).detach().requires_grad_(old_requires_grad)
        print("inner cloned tensor : ", _tensor.requires_grad)
        return _tensor
    print("input requires_grad: ", inputs[0].requires_grad)
    return traverse_obj(inputs, inner_to_mem_format)


super_run = torch._dynamo.optimize("aot_eager")(to_mem_format)

inputs = (torch.randn(4, 3, 4, 4).requires_grad_(True),)

output = super_run(torch.channels_last, inputs)

print("output requires_grad: ", output[0].requires_grad)