import torch.nn as nn

import onnxruntime
from torch import nn
import torch
import torch.nn.functional as F

print(torch.__version__)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.sftt = nn.Conv1d(1, 258, 256, bias=False, stride=80)

	def forward(self, x):
		pad = 5
		padded_signal = F.pad(x, (0, pad), mode='constant', value=0)
		stft_res = self.sftt(padded_signal.unsqueeze(dim=1))
		real_squared, imag_squared = (stft_res * stft_res).split(129, dim=1)
		return dict(o=real_squared)


model = Model()
x = torch.rand((2, 1024))
print(x.shape)
print(model(x))

torch.onnx.export(
		model, (x,),
		'fp32_opset_13.onnx',
		verbose=False,
		opset_version=13,
		export_params=True,
		do_constant_folding=True,
		input_names=['x']
)