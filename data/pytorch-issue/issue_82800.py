import torch

In [201]: x = torch.tensor([ 0.0552,  0.9730,  0.3973, -1.0780])

In [202]: torch.fake_quantize_per_tensor_affine(x, 0.1, 0, 0, 255)
Out[202]: tensor([0.1000, 1.0000, 0.4000, 0.0000])

In [203]: torch.fake_quantize_per_tensor_affine(x, torch.tensor(0.1), torch.tensor(0), 0, 255)
Out[203]: tensor([0.1000, 1.0000, 0.4000, 0.0000])