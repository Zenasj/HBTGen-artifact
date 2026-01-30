import torch.nn as nn
import torchvision

import os
import torch
import onnx
from model import create_model
from torch_utils import utils
from ptflops import get_model_complexity_info
from onnxruntime.quantization import quantize_dynamic, QuantType
from config import NUM_CLASSES, DEVICE, RESIZE_WIDTH, RESIZE_HEIGHT, OUT_DIR

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=NUM_CLASSES)
model_path = os.path.join(OUT_DIR, "best_model.pth")
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model_int8 = torch.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
model_int8.eval()

path_quant_onnx = os.path.join(OUT_DIR, "quant_onnx_model.onnx")
x_input = torch.randn(1, 3, RESIZE_HEIGHT, RESIZE_WIDTH)

torch.onnx.export(model_int8,               # model being run
                  x_input,                  # model input (or a tuple for multiple inputs)
                  path_quant_onnx,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])

qmodel_traced = torch.jit.trace(model_int8, example_inputs = x_input)