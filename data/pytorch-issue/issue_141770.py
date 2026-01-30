import torch.nn as nn

import torch

class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x,y):
		out = torch.meshgrid(x, y, indexing="xy")
		return out

onnx_program = torch.onnx.export(Model(), (torch.tensor(20),torch.tensor(20)), dynamo=True)