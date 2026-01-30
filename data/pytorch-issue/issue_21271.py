import torch.nn as nn
import random

import torch
from torch import nn
import numpy as np
import onnx
import onnxruntime as rt


shape = (2, 16, 96, 96)

def generate_model():
    net = nn.PReLU(16)
    model_name = 'only_relu.onnx'
    dummy_input = torch.randn(*shape)
    torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
    model = onnx.load(model_name)
    return model


def forward(model, inputs):
    sess = rt.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in sess.get_outputs()]
    res = dict(zip(outputs, sess.run(outputs, inputs)))
    return res


forward(generate_model(), {'input': np.random.rand(*shape).astype(np.float32)})