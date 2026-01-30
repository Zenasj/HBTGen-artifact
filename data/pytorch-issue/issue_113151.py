import torch

def forward(x,y):
  return torch.asinh(out=x, input=y)        
x1 = torch.rand([10],dtype=torch.float32)
x2 = x1.clone()
y = torch.tensor([487875.875, -956238.8125, 630736.0, -161079.578125, 104060.9375, 757224.3125, -153601.859375, -648042.5, 733955.4375, -214764.90625],dtype=torch.float32)
# we tried to generate y randomly as this shape and dtype but no bug was triggered
no_op_info = forward(x1,y)# result of eagermode
op_info = torch.compile(forward, mode='max-autotune-no-cudagraphs',fullgraph=True)(x2,y)# result of optimized mode
same_val = torch.allclose(no_op_info.to('cpu'), 
                          op_info.to('cpu'), 
                          rtol=1e-3, atol=1e-3, 
                          equal_nan=True)
if same_val == False : 
      raise ValueError('diff value')

import torch

def forward(y):
    return torch.asinh(y)

x1 = torch.rand([10],dtype=torch.float32)
x2 = x1
y = torch.tensor([487875.875, -956238.8125, 630736.0, -161079.578125, 104060.9375, 757224.3125, -153601.859375, -648042.5, 733955.4375, -214764.90625],dtype=torch.float32)
# we tried to generate y randomly as this shape and dtype but no bug was triggered
no_op_info = forward(y)  # result of eager mode
op_info = torch.compile(forward, mode='max-autotune-no-cudagraphs',fullgraph=True)(y)  # result of optimized mode
same_val = torch.allclose(no_op_info.to('cpu'),
                          op_info.to('cpu'),
                          rtol=1e-3, atol=1e-3,
                          equal_nan=True)

print(f"eager out: {op_info} \ncompiled out: {no_op_info}")

if same_val == False :
    raise ValueError(f'diff value')