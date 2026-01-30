import torch
import torch.nn as nn
import traceback

def forward(x, device):
  x = torch.diag_embed(input=x, dim1=-1,dim2=0,offset=6)        
  return x
input_tensor = torch.rand([6 ,8], dtype=torch.float32).to('cpu')
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

def forward(x, device):
  x = torch.diag_embed(input=x, dim1=-1,dim2=0,offset=6)
  return x
input_tensor = torch.rand([6 ,8], dtype=torch.float32).to('cuda')

no_op_info = forward(input_tensor, 'cpu')
print("build succeded")
op_info = torch.compile(forward, backend="aot_eager_decomp_partition",fullgraph=False,dynamic=True)(input_tensor, 'cuda')

same_val = torch.testing.assert_close(no_op_info.to('cpu'),
                        op_info.to('cpu'),
                        rtol=1e-3, atol=1e-3,
                        equal_nan=True)