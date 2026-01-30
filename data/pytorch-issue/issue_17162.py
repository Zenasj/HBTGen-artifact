import torch.onnx
from torch.autograd import Variable
import torch.onnx as torch_onnx
import torch.nn as nn

class TestONNX(nn.Module):
    def __init__(self):
        super(TestONNX, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x


TestONNX_c = TestONNX()
dummy_input = Variable(torch.randn(1, 3, 256, 256))
output = torch_onnx.export(TestONNX_c,
                           dummy_input,
                           'test.onnx',
                           verbose=True)

import torch.onnx
import torchvision
dummy_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.resnet50(pretrained=False) 
torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True)

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def GiB(val):
    return val * 1 << 30

def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            # print(model.read())
            parser.parse(model.read())
        return builder.build_cuda_engine(network)

build_engine_onnx('resnet50.onnx')

self.bn = nn.BatchNorm2d(10)
self.bn.num_batches_tracked = torch.Tensor(1)