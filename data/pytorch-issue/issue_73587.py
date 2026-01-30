import torch
torch.use_deterministic_algorithms(True)

x = torch.zeros((3,4,4)).float().cuda()

# Each one of these fail
x[torch.arange(2)] = 1.
# or x[torch.arange(2), 0] = 1.
# or x[range(2), 0] = 1.


""" Output:
RuntimeError: linearIndex.numel()*sliceSize*nElemBefore == value.numel()INTERNAL 
ASSERT FAILED at "C:\\cb\\pytorch_1000000000000\\work\\aten\\src\\ATen\\native\\cuda\\Indexing.cu":250,
please report a bug to PyTorch.
number of flattened indices did not match number of elements in the value tensor21
"""