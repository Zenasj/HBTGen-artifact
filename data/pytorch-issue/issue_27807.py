import torch
import torch.nn as nn

# takes seconds with CUDA 10.0 and minutes with CUDA 10.1
torch.zeros(25000, 300, device=torch.device("cuda"))

print("Pytorch version is:", torch.__version__)
print("Cuda version is:", torch.version.cuda)
print("cuDNN version is :", torch.backends.cudnn.version())
print("Arch version is :", torch._C._cuda_getArchFlags())