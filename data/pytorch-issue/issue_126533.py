import torch
torch.set_default_device('cuda')
print(torch.zeros([]).device)  # shows CUDA device
print(torch.Tensor().device)   # shows CPU device
torch.set_default_tensor_type(torch.cuda.FloatTensor)  # shows deprecation warning, but still works
print(torch.Tensor().device)   # shows CUDA device now