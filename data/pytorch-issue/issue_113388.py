import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_features = 25000
in_features = 10000
weight = nn.Parameter(torch.FloatTensor(out_features, in_features)).cuda()
nn.init.xavier_uniform_(weight)
batch_size = 200000
input = nn.Parameter(torch.FloatTensor(batch_size, in_features)).cuda()
nn.init.xavier_uniform_(input)
y = F.linear(input, weight).to(device, dtype=torch.half, non_blocking=True)
print(y.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_features = 25000
in_features = 10000
weight = nn.Parameter(torch.HalfTensor(out_features, in_features)).cuda()
nn.init.xavier_uniform_(weight)
batch_size = 200000
input = nn.Parameter(torch.HalfTensor(batch_size, in_features)).cuda()
nn.init.xavier_uniform_(input)
y = F.linear(input, weight).to(device, dtype=torch.half, non_blocking=True)
print(y.shape)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True