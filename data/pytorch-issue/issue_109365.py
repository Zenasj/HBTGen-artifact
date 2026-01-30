import torch
import torch.nn as nn
import os
from typing import List
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(gm.code)
    print(gm.graph)
    gm.graph.print_tabular()
    return gm.forward

class OpWrapperModule(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, inputs):
        result = self.op(**inputs)
        return result

def test_dynamic_shape_pow(is_dyn):
    print("Starting the test.................")
    sizes = [[340695, 80],
             [340658, 80],
             [340688, 80],
             [340658, 80],
             [340663, 80]]

    exponent = 2
    dev = torch.device("cpu")
    dtype = torch.float32
    op = torch.pow
    input_tensors = []
    for s in sizes:
        input_tensors.append(torch.rand(s, dtype=dtype))

    model = OpWrapperModule(op)
    model = torch.compile(model, backend=my_compiler, dynamic=is_dyn)
    model.to(dev)
    for s in input_tensors:
        s = s.to(dev)
        inputs = {"input":s, "exponent": exponent}
        result = model(inputs)

dyn_shape = os.getenv("DYNAMIC_SHAPE")
if dyn_shape is not None and dyn_shape == "0":
    test_dynamic_shape_pow(False)
else:
    test_dynamic_shape_pow(True)