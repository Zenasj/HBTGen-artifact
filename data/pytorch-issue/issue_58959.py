py
#!/usr/bin/env python3
import torch
asdf = torch.tensor([[1, 2], [3, 4]])
print('cpu')
print(asdf[:, -1])
print(asdf[:, -1].unique())
asdf = asdf.to(device='cuda')
print('gpu')
print(asdf[:, -1])
print(asdf[:, -1].unique())
qwer = asdf[:, -1]
print(qwer)
print(qwer.unique())

cpu
tensor([2, 4])
tensor([2, 4])
gpu
tensor([2, 4], device='cuda:0')
tensor([2, 3], device='cuda:0')
tensor([2, 4], device='cuda:0')
tensor([2, 3], device='cuda:0')

cpu
tensor([2, 4])
tensor([2, 4])
gpu
tensor([2, 4], device='cuda:0')
tensor([2, 4], device='cuda:0')
tensor([2, 4], device='cuda:0')
tensor([2, 4], device='cuda:0')