import torch

x = torch.nested_tensor((torch.randn(3, 4), torch.randn(3, 5)))
print(x[0])  # RuntimeError: Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor