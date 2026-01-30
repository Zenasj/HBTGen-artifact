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
        conv = self.conv(x)
        conv = conv * 0.5
        relu = self.relu(conv)
        return relu

model = MyModule().eval().cuda()
input = torch.randn((1, 3, 224, 224), dtype=torch.float32).cuda()
trt_ep = torch_tensorrt.compile(
        model, inputs=[input], ir="dynamo", debug=True, min_block_size=1,
        output_format="exported_program"
    )
torch.export.save(trt_ep, "/tmp/trt.ep")
ep = torch.export.load("/tmp/trt.ep")
gm = ep.module()