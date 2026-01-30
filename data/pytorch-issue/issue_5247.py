import torch

tensors = []
for i in range(1000):
    print(i)
    a = torch.Tensor(1000000000)
    a.share_memory_()
    tensors.append(a)