import torch
all0s = torch.zeros((32, ), dtype=torch.int64)
all0s.bincount()
# answer is tensor([32])
all0s = torch.zeros((32, 2), dtype=torch.int64)
all0s[:, 0].bincount()
# answer is tensor([31]). This is wrong! The same command also some times exits with segfault

all1s = torch.ones((32, 2), dtype=torch.int64)
all1s[:, 0].bincount()
# answer is tensor([ 2, 29]). Very wrong!

all0s[:, 0].clone().bincount()
# answer is tensor([32]). This is now correct!