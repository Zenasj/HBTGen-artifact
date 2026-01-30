import torch.nn as nn

import onnxruntime
from torch import nn
import torch

print(torch.__version__)

@torch.jit.script
def compute_output_lengths(x, lengths_fraction):
    return (lengths_fraction * x.shape[-1]).ceil().long()

@torch.jit.script
def temporal_mask(x, lengths):
    tup_prod = (1,) * (len(x.shape) - 2)
    return (torch.arange(x.shape[-1], dtype=lengths.dtype).unsqueeze(0) <
            lengths.unsqueeze(1)).view(x.shape[:1] + tup_prod + x.shape[-1:])

class Model(nn.Module):
    def forward(self, x, xlen):
        l = compute_output_lengths(x, xlen)
        tm = temporal_mask(x, l)
        return x * tm

model = Model()

x = torch.rand(2, 10)
xlen = torch.rand(2)

print(model(x, xlen))

torch.onnx.export(
        model, (x, xlen),
        'fp32_jit_prim_dtype_repro.onnx',
        verbose=True,
        opset_version=16,
        input_names=['x', 'xlen']
)

runtime = onnxruntime.InferenceSession('fp32_jit_prim_dtype_repro.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpu().numpy())))