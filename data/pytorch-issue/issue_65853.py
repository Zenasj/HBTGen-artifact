import torch.nn as nn

import torch

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        
    def forward(self, input):
        return MyRelu.apply(input)
    
module = MyModule()
x = torch.randn(5, 5)
y = module(x)

torch.onnx.export(module, x, 'test.onnx', export_params=True, opset_version=13, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, do_constant_folding=False, verbose=False, input_names=['input'], output_names=['output'])

@staticmethod
def symbolic(ctx, input):
      return g.op("Clip", input, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))