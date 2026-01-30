import torch
t = torch.tensor([])
t.median()

import torch

func_cls=torch.nanmedian
t = torch.tensor([])

def test():
	func_cls(t)
test()