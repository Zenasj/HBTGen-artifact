import onnx
import torch
import torch.nn as nn

# x: (batch_size, N, 3, 3)
# y: (batch_size, 10_000, 3)
# output: (batch_size, N, 10_000, 3)

class MatMulModule(nn.Module):
    '''
        Version 1: using auto-broadcasting with torch.matmul().
        Dynamic shapes are not working properly.
    '''
    def __init__(self):
        super(MatMulModule, self).__init__()

    def forward(self, x, y):
        x = x[:, :, None, :, :]
        y = y[:, None, :, :, None]
        return torch.matmul(x, y)

class BatchMatMulModule(nn.Module):
    '''
        Version 2: using torch.bmm() with manual broadcasting.
        Dynamic shapes are working properly.
    '''
    def __init__(self):
        super(BatchMatMulModule, self).__init__()

    def forward(self, x, y):
        bs = x.shape[0]
        N = x.shape[1]
        N2 = y.shape[1]

        # Manually broadcast & reshape for torch.bmm()
        x = x[:, :, None, :, :].repeat(1, 1, N2, 1, 1).reshape(bs * N * N2, 3, 3)
        y = y[:, None, :, :, None].repeat(1, N, 1, 1, 1).reshape(bs * N * N2, 3, 1)
        return torch.bmm(x, y).reshape(bs, N, N2, 3)

# Create model instance
# model = BatchMatMulModule() # Works fine
model = MatMulModule() # Doesn't work

# Example inputs
x = torch.randn(1, 5, 3, 3)
y = torch.randn(1, 10_000, 3)

# Dynamic shapes
dyn_name = 'dim1'
dyn_min = 1
dyn_max = 10
dynamic_shapes = {}
dynamic_shapes['x'] = {1: torch.export.Dim(dyn_name, min=dyn_min, max=dyn_max)}
dynamic_shapes['y'] = {0: torch.export.Dim.STATIC}

# Export
ep = torch.export.export(model, args=tuple(), kwargs={'x': x, 'y': y}, dynamic_shapes=dynamic_shapes)
onnx_program = torch.onnx.export(
    ep,
    dynamo=True,
    optimize=True,
)

onnx_program.save('matmul.onnx')

# Check input shapes
print("Input shapes for first arg:")
m = onnx.load('matmul.onnx')
print(m.graph.input[0])