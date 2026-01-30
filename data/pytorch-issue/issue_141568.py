import torch.nn as nn

import torchvision
import torch

class M(torch.nn.Module):
	def __init__(self):
		super().__init__()
		pass

	def forward(self, x, count):
		out = torchvision.ops.batched_nms(x[0], x[1], x[2], x[3])
		return out

model = M()
torch.manual_seed(1)
args = (
	(
		torch.rand(20, 4, dtype=torch.float),
		torch.rand(20, dtype=torch.float),
		torch.randint(0, 2, (20,), dtype=torch.float),
		0,
	),
	3,
)
torch.onnx.export(model, args, dynamo=True)

#torch.onnx.dynamo_export(model, *args)