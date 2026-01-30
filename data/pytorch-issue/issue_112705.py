import torch
import torch.nn as nn
import traceback

def forward(x, device):       
  x = torch.diff(prepend=x, append=torch.rand([3], dtype=torch.float32).to('cpu'),dim=0,input=torch.rand([3], dtype=torch.float32).to('cpu'),n=5)        
  return x
inf = float('inf')
nan = float('nan')
is_valid = True
input_tensor = torch.rand([3], dtype=torch.float32).to('cpu')
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
import torch.nn as nn
import traceback

def forward(x, device):       
    x = torch.diff(
        prepend=x,
        append=torch.tensor([1, 2, 3], dtype=torch.float32).to(device),
        dim=0,
        input=torch.tensor([4, 5, 6], dtype=torch.float32).to(device),
        n=5
    )        
    return x
input_tensor = torch.rand([3], dtype=torch.float32).to('cpu')

cuda_tensor = input_tensor.clone().detach().to('cuda')
no_op_info = forward(input_tensor, 'cpu')
print("build succeded")

op_info = torch.compile(forward, fullgraph=True)(cuda_tensor, 'cuda')

print("Results", forward(input_tensor, "cpu"), no_op_info, op_info)

same_val = torch.allclose(
    no_op_info.to('cpu'), 
    op_info.to('cpu'), 
    rtol=1e-3,
    atol=1e-3, 
    equal_nan=True
)

if same_val == False: 
    print("BUGBUG DIFFERENTIAL")
    raise ValueError(f'diff value {no_op_info}, {op_info}')
else:
    print("no_error")