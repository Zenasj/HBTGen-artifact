import torch

def forward(x):
  return torch.unfold_copy(dimension=1, input=x,size=0,step=7) 

x = torch.rand([1,0], dtype=torch.float32)# generate arg
forward(x)# on eagermode
print("build succeeded")
torch.compile(forward, mode='max-autotune',fullgraph=True)(x)# encountered a ZeroDivisionError on torch.compile mode