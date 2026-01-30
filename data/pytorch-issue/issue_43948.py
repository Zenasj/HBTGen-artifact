import torch
import torch.nn as nn

N, C, L = 2, 4, 5
if False:
    # example from: https://github.com/pytorch/pytorch/issues/34002#issuecomment-656769904
    class MyClass(torch.nn.Module):
        def __init__(self):
            super(MyClass, self).__init__()
            self.num_batches_tracked = 0
        def forward(self, x):
            self.num_batches_tracked += 1
            return x

else:
    # example using BN
    class MyClass(torch.nn.Module):
        def __init__(self):
            super(MyClass, self).__init__()
            self.bn = torch.nn.BatchNorm1d( C )
        def forward(self, x):
            return self.bn(x)

model = MyClass()
model.eval()
x_in = torch.zeros((N, C, L))
traced_model = torch.jit.trace(model, x_in)
scripted_model = torch.jit.script(model)

# ONNX export
print('ONNX export of plain model...')
torch.onnx.export(model, x_in, 'f.onnx', example_outputs=x_in)  # => OK

print('ONNX export of scripted model...')
torch.onnx.export(scripted_model, x_in, 'f.onnx', example_outputs=x_in) #  => FAIL

# CoreML export
import coremltools as ct

print('CoreML export of plain model...')
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=x_in.shape)]
)  # => OK

print('CoreML export of scripted model...')
coreml_model = ct.convert(
    scripted_model,
    inputs=[ct.TensorType(shape=x_in.shape)]
)  #  => FAIL