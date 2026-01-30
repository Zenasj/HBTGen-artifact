import torch
print(torch.__version__)

a = torch.tensor([[float("inf")+0.0000j], [-1.-8.7423e-08j]])
res = torch.exp(a)
print(f"{res=}")