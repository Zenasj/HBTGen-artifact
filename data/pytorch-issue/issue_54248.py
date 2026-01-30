import torch

x = torch.randn(4, 5, dtype=torch.float32)
mkldnn_x = x.to_mkldnn()
mkldnn_y = torch.randn(4, 5, dtype=torch.float32).to_mkldnn()
mkldnn_y.copy_(mkldnn_x)

print(x)
print(mkldnn_y.to_dense())