import torch.nn as nn

import torch
import torch.nn.attention.flex_attention
torch.set_default_device("cuda")
print(torch.__version__)
flex_compiled = torch.compile(torch.nn.attention.flex_attention.flex_attention)
for fix_issue in [False, True]:
	for i in range(10):
		torch.manual_seed(0)
		shape = (1, 16, 4096, 64)
		Q = torch.randn(shape, requires_grad=True)
		K = torch.randn(shape, requires_grad=True)
		V = torch.randn(shape, requires_grad=True)
		flex_compiled(Q, K, V) # why does this line have to be here??

		K_sliced = K[:, :, :-128]
		V_sliced = V[:, :, :-128]

		if fix_issue:
			K_sliced = K_sliced.clone()

		flex_compiled(Q, K_sliced, V_sliced).sum().backward()
		print("Q", Q.grad.mean(), "K", K.grad.mean(), "V", V.grad.mean(), K_sliced.is_contiguous())