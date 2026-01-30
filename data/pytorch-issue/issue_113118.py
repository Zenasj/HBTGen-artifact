import torch

def forward(x):
  return torch.unique_consecutive(dim=0,input=x)        

x = torch.rand([2],dtype=torch.float32)# generate arg
forward(x)# on eagermode
print("build succeeded")
torch.compile(forward, fullgraph=True)(x)# on torch.compile mode(with fullgrah=True)

import torch
import torch.onnx
torch.onnx.dynamo_export(lambda x: torch.unique(x), torch.arange(10))