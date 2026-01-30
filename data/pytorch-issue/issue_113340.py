import torch.nn as nn

import torch


def kernel1(tensor):
    return tensor + 2


def dispatcher1(input):
    kernel = get_kernel(dispatcher1, type(input))
    return kernel(input)


def kernel2(tensor):
    return tensor - 2


def dispatcher2(input):
    kernel = get_kernel(dispatcher2, type(input))
    return kernel(input)


# We actually use the function and type as keys, rather than their names.
# However, this currently not supported, but should be easy to add after
# https://github.com/pytorch/pytorch/pull/111196
REGISTRY = {
    "dispatcher1": {"Tensor": kernel1},
    "dispatcher2": {"Tensor": kernel2},
}


def get_kernel(dispatcher, input_type):
    dispatcher_registry = REGISTRY[dispatcher.__name__]
    for cls in input_type.__mro__:
        kernel = dispatcher_registry[cls.__name__]
        break
    return kernel

cfn = torch.compile(dispatcher1, fullgraph=True)
torch.testing.assert_close(int(cfn(torch.tensor(3))), 5)

cfn = torch.compile(dispatcher2, fullgraph=True)
torch.testing.assert_close(int(cfn(torch.tensor(3))), 1)

class Pipeline(torch.nn.Module):
    def forward(self, input):
        input = dispatcher1(input)
        input = dispatcher2(input)
        return input


cfn = torch.compile(Pipeline(), fullgraph=True)
torch.testing.assert_close(int(cfn(torch.tensor(3))), 3)