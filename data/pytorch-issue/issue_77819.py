import torch

M = torch.randn(10, 3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
print(torch.baddbmm(M, batch1, batch2).size())
# torch.Size([10, 3, 5])

M = torch.randn(10, 3, 5).to("mps")
batch1 = torch.randn(10, 3, 4).to("mps")
batch2 = torch.randn(10, 4, 5).to("mps")
print(torch.baddbmm(M, batch1, batch2).size())
#RuntimeError: input tensor does not match matmul output shape

torch.Size([10, 3, 5])
torch.Size([10, 3, 5])