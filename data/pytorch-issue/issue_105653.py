import torch.nn as nn

import torch
from torch import nn

from torch import _dynamo
from torch._functorch.aot_autograd import aot_module_simplified
import functorch
from functorch.compile import make_boxed_func

def graph_processing_pytorch(gm, example_inputs):
    # graph transform (graph optimization) for the captured graph in pytorch
    print("captured graph in pytorch")
    gm.print_readable()

def graph_processing_aot_forward(gm, example_inputs):
    # graph transform (graph optimization) for the captured graph in aot autograd forward graph
    print("captured graph in aot autograd forward")
    gm.print_readable()

def graph_processing_aot_backward(gm, example_inputs):
    # graph transform (graph optimization) for the captured graph in aot autograd backward graph
    print("captured graph in aot autograd backward")
    gm.print_readable()

def forward_compiler(gm, example_inputs):
    graph_processing_aot_forward(gm, example_inputs)
    return make_boxed_func(gm.forward)

def backward_compiler(gm, example_inputs):
    graph_processing_aot_backward(gm, example_inputs)
    return make_boxed_func(gm.forward)

def custom_backend(gm, example_inputs):
    graph_processing_pytorch(gm, example_inputs)
    return aot_module_simplified(
        gm, example_inputs,
        fw_compiler=forward_compiler,
        bw_compiler=backward_compiler
    )

class BackboneModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.bn1(self.drop1(x))
        return x


model = BackboneModel()

opt_model = torch.compile(model, backend=custom_backend)

input = torch.randn(64, 6, 32, 32)

output1 = opt_model(input)

# calling .eval in the whole model works
# opt_model.eval()

# calling .eval in sub model does not trigger re-compilation
opt_model.drop1.eval()
opt_model.bn1.eval()

output2 = opt_model(input)
output3 = opt_model(input)

print((output2 - output3).abs().max().item()) # huge number, nondeterministic!