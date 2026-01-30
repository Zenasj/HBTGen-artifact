import torch
import torch.cuda.comm
torch.cuda.comm.scatter(tensor=torch.tensor([1]), devices=[1], streams='')