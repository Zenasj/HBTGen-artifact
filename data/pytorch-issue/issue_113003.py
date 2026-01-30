import torch

def forward(x,y):
  return torch._C._linalg.linalg_matrix_power(input=x, n=2,out=y)        
x = torch.rand([2,2],dtype=torch.float32)
y = torch.rand([2,2],dtype=torch.float32)
forward(x,y)
print("build succeeded")
torch.compile(forward, mode='default',fullgraph=False,dynamic=True)(x,y)