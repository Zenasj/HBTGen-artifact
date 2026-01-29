# torch.rand(B, 3, 512, 512, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.pooling import _MaxUnpoolNd
from torch.nn.modules.utils import _pair

class MaxUnpool2dop(Function):
    """Wraps F.max_unpool2d with ONNX symbolic function for export."""
    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding, output_size):
        return F.max_unpool2d(input, indices, kernel_size, stride, padding, output_size)

    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride, padding, output_size):
        input_shape = g.op('Shape', input)
        const_0 = g.op('Constant', value_t=torch.tensor(0))
        const_1 = g.op('Constant', value_t=torch.tensor(1))
        output_size_list = list(output_size) if output_size else []
        const_size = g.op('Constant', value_t=torch.tensor(output_size_list)) if output_size else None

        batch_size = g.op('Gather', input_shape, const_0, axis_i=0)
        channel = g.op('Gather', input_shape, const_1, axis_i=0)

        # Calculate height and width for output_size
        height = g.op('Gather', input_shape, g.op('Constant', value_t=torch.tensor(2)), axis_i=0)
        height = g.op('Sub', height, const_1)
        height = g.op('Mul', height, g.op('Constant', value_t=torch.tensor(stride[1])))
        height = g.op('Add', height, g.op('Constant', value_t=torch.tensor(kernel_size[1])))

        width = g.op('Gather', input_shape, g.op('Constant', value_t=torch.tensor(3)), axis_i=0)
        width = g.op('Sub', width, const_1)
        width = g.op('Mul', width, g.op('Constant', value_t=torch.tensor(stride[0])))
        width = g.op('Add', width, g.op('Constant', value_t=torch.tensor(kernel_size[0])))

        channel_step = g.op('Mul', height, width)
        batch_step = g.op('Mul', channel_step, channel)

        range_channel = g.op('Range', const_0, channel, const_1)
        range_channel = g.op('Reshape', range_channel, g.op('Constant', value_t=torch.tensor([1, -1, 1, 1])))
        range_channel = g.op('Mul', range_channel, channel_step)
        range_channel = g.op('Cast', range_channel, to_i=7)

        range_batch = g.op('Range', const_0, batch_size, const_1)
        range_batch = g.op('Reshape', range_batch, g.op('Constant', value_t=torch.tensor([-1, 1, 1, 1])))
        range_batch = g.op('Mul', range_batch, batch_step)
        range_batch = g.op('Cast', range_batch, to_i=7)

        indices = g.op('Add', indices, range_channel)
        indices = g.op('Add', indices, range_batch)

        # Include output_size as input if provided
        inputs = [input, indices]
        if const_size is not None:
            inputs.append(const_size)
        
        return g.op('MaxUnpool', *inputs, kernel_shape_i=kernel_size, strides_i=stride)

class MaxUnpool2d(_MaxUnpoolNd):
    """Custom MaxUnpool2d for ONNX compatibility."""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        if isinstance(output_size, torch.Size):
            output_size = tuple(output_size)
        return MaxUnpool2dop.apply(input, indices, self.kernel_size, self.stride, self.padding, output_size)

class MyModel(nn.Module):
    """Simple model using custom MaxUnpool2d for testing ONNX export."""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_pooled, indices = self.pool(x)
        return self.unpool(x_pooled, indices)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 512, 512, dtype=torch.float32)

