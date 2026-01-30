import torch
import torch.nn as nn
import traceback
def forward(x, device):  
  x = torch.atan(out=x, input=torch.randint(-9223372036854775808, 9223372036854775807, [10,9,8,6], dtype=torch.int64).to('cpu'))              
  return x
input_tensor = torch.rand([10,9,8,6], dtype=torch.float32).to('cpu')
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