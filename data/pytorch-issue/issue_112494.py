import torch
import torch.nn as nn
import traceback

def forward(x, device):

  
  x = torch.var(out=x, correction=4,dim=0,input=torch.rand([], dtype=torch.float32).to('cpu'),keepdim=True)        
  return x

input_tensor = torch.rand([10, 9, 8], dtype=torch.float32).to('cpu')
cuda_tensor = input_tensor.clone().to('cuda')
no_op_info = forward(input_tensor, 'cpu')
print("build succeded")
op_info = torch.compile(forward, mode='reduce-overhead',fullgraph=True,dynamic=True)(cuda_tensor, 'cuda')

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
import torch.nn as nn
import traceback

def fn(x, device):
  x = torch.var(correction=4, dim=0, input=x, keepdim=True, out=torch.rand_like(x))
  return x

cuda_tensor = torch.rand([10, 9, 8], dtype=torch.float32, device='cuda')
fn(cuda_tensor, 'cuda')
op_info = torch.compile(fn, dynamic=True)(cuda_tensor, 'cuda')