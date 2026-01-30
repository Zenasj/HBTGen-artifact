import torch

nested_tensor1.shape = [torch.Size([2, 4, 8]), torch.Size([2, 6, 8])]
nested_tensor2.shape = [torch.Size([2, 8, 8]), torch.Size([2, 8, 8])]

nested_tensor1.shape = [torch.Size([2, 4, 8]), torch.Size([2, 6, 8])]
nested_tensor2.shape = [torch.Size([2, 8, 8]), torch.Size([2, 8, 8])]

inputs = [torch.randn(5, 16), torch.randn(3, 32)]
others = [torch.randn(16, 32), torch.randn(32, 64)]