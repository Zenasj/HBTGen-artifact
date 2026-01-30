import torch
print(torch.__version__)    #  1.13.0.dev20220913+cpu
torch.floor_divide(input=torch.tensor([1, 1]), other=torch.tensor([1]), out=torch.tensor([[[1],[1]]]))

torch.floor_divide(input=-1, other=torch.tensor([1]), out=torch.ones([1,0,1]))
torch.fmod(input=torch.tensor([1, 1]), other=torch.tensor([1]), out=torch.tensor([[[1],[1]]]))
torch.fmod(input=-1, other=torch.tensor([1]), out=torch.ones([1,0,1]))