import torch.nn as nn

import onnx
import torch

class M(torch.nn.Module):     
    def __init__(self, hidden_size):
        super().__init__()
        self.weight_ih = torch.nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size)
        )
        
    def forward(self, x):
        gates = torch.mm(x, self.weight_ih.t())
        # matmul works
        #gates = x.matmul(self.weight_ih.t())
        return gates
    
m = M(10)
m.eval()
# issue occurs in both jitted and non-jitted case
m = torch.jit.script(m)

x = torch.ones(1, 10)
output = m(x)

torch.onnx.export(
    m,
    x,
    "test.onnx",
    example_outputs=output,
    input_names="x",
    opset_version=11
)

onnx_model = onnx.load("test.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))