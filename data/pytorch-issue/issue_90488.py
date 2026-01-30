# sample code to reproduce the issue
import torch
old = torch.tensor([0,1,2,3,4,5,6,7,8,9])
new = old[0:6]
print("Input:",new)
print("Output:\n",torch.as_strided(input=new,size=(6,4),stride=(1,1)))