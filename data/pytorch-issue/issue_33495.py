import torch.nn as nn

import torch
from typing import Tuple
from torch import Tensor
from copy import deepcopy

HIDDEN_SIZE = 4
NUM_LAYERS = 1

class SimpleControlFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    def forward(self, x: Tensor, hn: Tensor, cn: Tensor, 
                sos: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        out, (hn, cn) = self.lstm(x, (hn, cn))
        if sos:
            out += 1.
        return out, hn, cn

def gen_args(batch, seq_len, sos, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
    """Generates input args for SimpleControlFlow."""
    y = torch.randn(seq_len, batch, hidden_size)
    zeros = torch.zeros(num_layers, batch, hidden_size) # hn == cn == zeros
    return y, zeros, zeros, torch.tensor([sos])

model = SimpleControlFlow()

args1 = gen_args(batch=1, seq_len=3, sos=False)
args2 = gen_args(batch=1, seq_len=1, sos=True)
model_t = torch.jit.trace(deepcopy(model), args1, check_trace=True)
out1_nn = model(*args1)
out1_trace = model_t(*args1)
out2_nn = model(*args2)
out2_trace = model_t(*args2)

assert torch.allclose(out1_nn[0], out1_trace[0]), "Will pass as args1 used for tracing.\n"
assert not torch.allclose(out2_nn[0], out2_trace[0]), "Will pass as `sos` was traced as constant = False.\n"

args = gen_args(batch=1, seq_len=3, sos=False)

model = torch.jit.script(model)

example_outputs = model(*args) # i.e. this runs 


input_names = ['y', 'hn_in', 'cn_in', 'sos']
output_names = ['out', 'hn_out', 'cn_out']
dynamic_axes = {
    "y": {0: "seq_len", 1: "batch"},
    "hn_in": {1: "batch"},
    "cn_in": {1: "batch"},
    "out": {0: "seq_len", 1: "batch"},
    "hn_out": {1: "batch"},
    "cn_out": {1: "batch"},
}

torch.onnx.export(
    model,
    args,
    'model.onnx',
    export_params=True,
    verbose=True,
    example_outputs=example_outputs,
    dynamic_axes=None,
    input_names=input_names,
    output_names=output_names,
    opset_version=11,
    )