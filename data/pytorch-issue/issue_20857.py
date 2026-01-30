import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def GiB(val):
    return val * 1 << 30

def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)

build_engine_onnx("alexnet.onnx")

import tensorrt as trt
import torch
print("tensorrt version:", trt.__version__, "torch version:", torch.__version__)
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def GiB(val):
    return val * 1 << 30

def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = GiB(1)
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
        return builder.build_engine(network, config)

build_engine_onnx("alexnet.onnx")