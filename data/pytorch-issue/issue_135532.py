import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.empty_like(x)

device = torch.device('cuda')
model = Model().to(device)
x = torch.rand(1024, 20, 16).to(device)

""""                                                                                                                                             
Fails with:                                                                                                                                      
## Exception summary                                                                                                                             
                                                                                                                                                 
<class 'TypeError'>: aten_empty_like() got an unexpected keyword argument 'pin_memory'                                                           
⬆️                                                                                                                                                
<class 'torch.onnx._internal.exporter._errors.GraphConstructionError'>: Error when calling function 'TracedOnnxFunction(<function aten_empty_lik\
e at 0x74e67b83fb50>)' with args '[SymbolicTensor('x', type=Tensor(FLOAT), shape=[1024,20,16], producer=None, index=None)]' and kwargs '{'pin_me\
mory': False}'                                                                                                                                   
⬆️                                                                                                                                                
<class 'torch.onnx._internal.exporter._errors.ConversionError'>: Error when translating node %empty_like : [num_users=1] = call_function[target=\
torch.ops.aten.empty_like.default](args = (%x,), kwargs = {pin_memory: False}). See the stack trace for more information.                        
"""
onnx_program = torch.onnx.export(model, x, "sort.onnx", dynamo=True, fallback=False)