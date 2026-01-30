import torch
results = dict()
input = torch.rand([10, 3, 5, 0], dtype=torch.float32)
batch1 = torch.rand([10, 3, 4], dtype=torch.float32)
batch2 = torch.rand([10, 4, 5], dtype=torch.float32)
input.clone().baddbmm(batch1, batch2)
# RuntimeError: expand(torch.FloatTensor{[10, 3, 5, 0]}, size=[10, 3, 5]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)
input.clone().baddbmm_(batch1, batch2)
# succeed