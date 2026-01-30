import torch

device = torch.device("cuda")
Tensor=torch.cuda.HalfTensor(30*1000*1000,128)
out=torch.mv(Tensor,Tensor[0])
print(out[0])