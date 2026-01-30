import torch.nn as nn

import onnxruntime
from torch import nn
import torch

print(torch.__version__)

@torch.jit.script
def compute_output_lengths(x, lengths_fraction):
	if lengths_fraction is None:
		return torch.full(x.shape[:1], x.shape[-1], device = x.device, dtype = torch.long)
	return (lengths_fraction * x.shape[-1]).ceil().long()

@torch.jit.script
def temporal_mask(x, lengths):
	return (torch.arange(x.shape[-1], device = x.device, dtype = torch.long).unsqueeze(0) <
			lengths.unsqueeze(1)).view(x.shape[:1] + (1, ) * (len(x.shape) - 2) + x.shape[-1:])

class Model(nn.Module):
	def forward(self, x, xlen):
		l = compute_output_lengths(x, xlen)
		tm = temporal_mask(x, l)
		x = x * tm
		return dict(o=x)

model = Model()
model.to(dtype=torch.float16, device='cuda')

x = torch.rand(2, 10).to(dtype=torch.float16, device='cuda')
xlen = torch.rand(2).to(dtype=torch.float16, device='cuda')

print(model(x, xlen))

torch.onnx.export(
		model, (x, xlen),
		'fp16_jit_repro.onnx',
		verbose=False,
		opset_version=12,
		export_params=None,
		do_constant_folding=True,
		input_names=['x', 'xlen']
)

print('runtime...')
runtime = onnxruntime.InferenceSession('fp16_jit_repro.onnx')
print(runtime.run(None, dict(x=x.cpu().numpy(), xlen=xlen.cpy().numpy())))