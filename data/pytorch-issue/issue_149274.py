import torch

x = torch.randn(2, 64, 32, 32).to_mkldnn()
out2 = torch.mkldnn_max_pool2d(x, kernel_size=3, stride=0)