import torch
import torch.nn as nn

x = torch.rand(70000, 1, 2).cuda()

bn = nn.BatchNorm1d(1)
bn.cuda()
bn.eval()

xbn = bn(x)
xbn.size()

# RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.