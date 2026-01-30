import torch

buffers = []
for i in range(2000):
    print(i)
    buffers.append(torch.empty(size=(1,)).share_memory_())