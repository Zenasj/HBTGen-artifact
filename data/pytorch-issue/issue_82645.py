import torch

x = torch.tensor(-1.8359375, requires_grad=True, dtype=torch.float16)
idx = torch.tensor([0])

out = torch.index_select(x, 0, idx)
out.backward()

x_mps = x.to("mps")
idx_mps = idx.to("mps")
out = torch.index_select(x_mps, 0, idx_mps)
out.backward()   # error: input types 'tensor<1xf16>' and 'tensor<1xf32>' are not broadcast compatible