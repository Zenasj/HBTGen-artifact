import torch
import torch.nn as nn
import traceback

def forward(x, device):

  
  x = torch.nn.functional.adaptive_max_pool3d_with_indices(output_size=x, input=torch.rand([9, 10, 9, 8, 6], dtype=torch.float32),return_indices=True)        
  return x
inf = float('inf')
nan = float('nan')
is_valid = True
input_tensor = 5
cuda_tensor = input_tensor
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