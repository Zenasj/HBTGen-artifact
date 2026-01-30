import torch
import torch.nn as nn

class MyMod(torch.nn.Module):
		def __init__(self):
				super().__init__()
				self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
				self.other_attr = torch.rand((2, 2))

		def __getattr__(self, name):
				custom_dict = self.custom_dict
				if name in custom_dict:
						return custom_dict[name]
				return super().__getattr__(name)

		def forward(self, x):
				return x @ self.other_attr + self.queue[-1]