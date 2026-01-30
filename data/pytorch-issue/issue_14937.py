import torch
print("torch version:", torch.__version__)

import torch.onnx as torch_onnx
import torch.nn as nn
import mxnet
print("mxnet version:", mxnet.__version__)
import mxnet.contrib.onnx as onnx_mxnet


class TestONNX(nn.Module):
    def __init__(self):
        super(TestONNX, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x


TestONNX_c = TestONNX()
dummy_input = torch.randn(1, 3, 256, 256)
torch_onnx.export(TestONNX_c,
                           dummy_input,
                           'test.onnx',
                           keep_initializers_as_inputs=True,
                           verbose=True)
sym, arg, aux = onnx_mxnet.import_model("test.onnx")