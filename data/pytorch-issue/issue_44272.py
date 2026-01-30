import torch
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
tt = torch.Tensor([1])
torch.exp(1j*tt)