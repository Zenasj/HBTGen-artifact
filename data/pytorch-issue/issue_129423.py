import torch.nn as nn

py
import torch
import torch_tensorrt 

class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            return out

model = MyModule().eval().cuda()

compile_spec = {
    "device": torch_tensorrt.Device("cuda:0"),
    "enabled_precisions": {torch.float},
    "ir": "torch_compile",
    "min_block_size": 1,
    "debug": True
}

input_bs4 = torch.randn((4, 3, 224, 224)).to("cuda")
torch._dynamo.mark_dynamic(input_bs4, 0, min=2, max=8)
# Compile the model
trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
out_bs4 = trt_model(input_bs4)
input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
out_bs6 = trt_model(input_bs6)