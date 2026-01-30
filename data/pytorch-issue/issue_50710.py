import torch

with unittest.mock.patch.object(torch.Tensor, "__getitem__"):
    pass  # do unrelated stuff with a patched __getitem__
# back to normal (except quantization is broken)
assert torch.Tensor.__getitem__ is torch._C._TensorBase.__getitem__  # passes

torch.Tensor.__getitem__ = None
del torch.Tensor.__getitem__
assert torch.Tensor.__getitem__ is torch._C._TensorBase.__getitem__  # passes