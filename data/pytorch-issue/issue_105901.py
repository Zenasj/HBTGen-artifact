import torch.nn as nn

import torch
import torch._dynamo as torchdynamo
import copy

class ConvMaxpool2d(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, bias=True)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))

def run_max_pool2d():
    batch_size = 116
    model = ConvMaxpool2d().eval()
    x = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        print("exported_model is: {}".format(exported_model), flush=True)

if __name__ == "__main__":
    run_max_pool2d()

def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    convolution_default = torch.ops.aten.convolution.default(arg0, _param_constant0, _param_constant1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg0 = _param_constant0 = _param_constant1 = None
    relu_default = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
    max_pool2d_with_indices_default = torch.ops.aten.max_pool2d_with_indices.default(relu_default, [3, 3], [2, 2]);  relu_default = None
    getitem = max_pool2d_with_indices_default[0]
    getitem_1 = max_pool2d_with_indices_default[1];  max_pool2d_with_indices_default = None
    return pytree.tree_unflatten([getitem], self._out_spec)