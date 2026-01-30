import torch

class H(torch.jit.ScriptModule):
	def __init__(self):
		super(H, self).__init__()

	@torch.jit.script_method
	def forward(self):
		r"""docstring"""
		return 1

print('H', H.forward, H.forward.__doc__)