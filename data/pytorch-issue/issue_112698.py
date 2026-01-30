import torch
import torch.nn as nn
import traceback

def forward(x, device):
  x = torch.special.i0(input=x, out=torch.rand([10, 9, 8], dtype=torch.float32).to('cpu'))        
  return x

input_tensor = torch.rand([10, 9, 8], dtype=torch.float32).to('cpu')
cuda_tensor = input_tensor.clone().to('cuda')
no_op_info = forward(input_tensor, 'cpu')
print("build succeded")
op_info = torch.compile(forward, mode='max-autotune',fullgraph=False,dynamic=True)(cuda_tensor, 'cuda')

same_val = torch.allclose(no_op_info.to('cpu'), 
                        op_info.to('cpu'), 
                        rtol=1e-3, atol=1e-3, 
                        equal_nan=True)
if same_val == False : 
    print("BUGBUG DIFFERENTIAL")
    raise ValueError('diff value')
else :
    print("no_error")

import torch

def forward(x):
  x = torch.special.i0(input=x)
  return x

input_tensor = torch.rand([10, 9, 8], dtype=torch.float32, device="cuda")

torch.compile(forward, dynamic=True)(input_tensor)

def forward(x, device):
  x = torch.special.i0(input=x, out=torch.rand([10, 9, 8], dtype=torch.float32).to('cpu'))        
  return x