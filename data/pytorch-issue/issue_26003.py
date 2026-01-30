import torch
t1 = torch.rand((2,3), device='cpu')  # create CPU tensor
t1.put_(torch.tensor([1, 30]), torch.tensor([9., 10.]))  # yields "RuntimeError: invalid argument 2: out of range: 20 out of 6 at /Users/administrator/nightlies/pytorch-1.0.0/wheel_build_dirs/wheel_3.7/pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:286"
t1.put_(torch.tensor([1, 3]), torch.tensor([9., 10.]))  # works
t2 = torch.rand((2,3), device='cuda')  # create CUDA tensor
t2.put_(torch.tensor([1, 30], device='cuda'), torch.tensor([9., 10.], device='cuda'))  # yields the errors pasted in the next block of code
t2.put_(torch.tensor([1, 3], device='cuda'), torch.tensor([9., 10.], device='cuda'))  # yields "RuntimeError: CUDA error: device-side assert triggered". We need to exit IPython and restart it to perform any CUDA operation now.