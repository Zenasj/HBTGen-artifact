import os

import torch
from yolov7 import create_yolov7_model

model = create_yolov7_model(architecture="yolov7")

input = torch.randn((1, 3, 640, 640))

dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
torch.onnx.export(
    model,
    input,
    os.path.join(f"output.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes=dynamic_axes,
)