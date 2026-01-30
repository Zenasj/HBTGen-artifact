import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

@torch.jit.script
def roi_pooling(x):
    # type: (Tensor) -> List[Tensor]
    out = []
    for i in range(1):
        out.append(x)
    return out

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.adaptive_avg_pool = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(512, 50),
            nn.ReLU(),
            )
    def forward(self, x):
        out = roi_pooling(x)
        out = [self.adaptive_avg_pool(x) for x in out]
        output_concat = torch.cat(out, 0)
        return output_concat

x = torch.randn(1, 512, 10, 10)

model = MyModule()
torch.onnx.export(model,               # model being run
                   x,                         # model input (or a tuple for multiple inputs)
                  "final.onnx",   # where to save the model (can be a file or file-like object)
                  #export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12, verbose=True,          # the ONNX version to export the model to
                  do_constant_folding=True)

import torch
import torch.nn as nn
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

@torch.jit.script
def roi_pooling(x):
    # type: (Tensor) -> (Tensor)
    return x

class RoiPooling(torch.jit.ScriptModule):
    def __init__(self):
        super(RoiPooling, self).__init__()
        self.adaptive_avg_pool = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(512, 50),
            nn.ReLU(),
            )
    @torch.jit.script_method
    def forward(self, x):
        out = roi_pooling(x)
        out = self.adaptive_avg_pool(out)
        for i in range(1, 2):
            x_out = roi_pooling(x)
            x_out = self.adaptive_avg_pool(x)
            out = torch.cat([out, x_out], 0)
        return out

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.roi_pool = RoiPooling()
    def forward(self, x):
        out = self.roi_pool(x)
        return out

x = torch.randn(1, 512, 10, 10)
model = torch.jit.trace(MyModule(), x)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "final.onnx",   # where to save the model (can be a file or file-like object)
                  #export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11, verbose=True,          # the ONNX version to export the model to
                  do_constant_folding=True,
                  example_outputs=model(x))