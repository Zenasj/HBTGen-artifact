import torch

torch.ops.load_library("path/to/build/libnumpy_test.dylib")

a = torch.rand(5, 5)
torch.ops.test_ops.numpy_test(a)