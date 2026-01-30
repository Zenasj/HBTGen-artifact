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
	return (torch.arange(x.shape[-1], device = x.device, dtype = lengths.dtype).unsqueeze(0) <
			lengths.unsqueeze(1)).view(x.shape[:1] + (1, ) * (len(x.shape) - 2) + x.shape[-1:])

class Model(nn.Module):
	def forward(self, x, xlen):
		l = compute_output_lengths(x, xlen)
		tm = temporal_mask(x, l)
		x = x * tm
		return dict(o=x)

model = Model()
model.to(device='cuda')

x = torch.rand(2, 10).to( device='cuda')
xlen = torch.rand(2).to( device='cuda')

print(model(x, xlen))

torch.onnx.export(
		model, (x, xlen),
		'fp32_jit_prim_dtype_repro.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'xlen']
)

runtime = onnxruntime.InferenceSession('fp32_jit_prim_dtype_repro.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpu().numpy())))