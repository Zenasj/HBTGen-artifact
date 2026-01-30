Upsample

pytorch

tensorflow

import torch
import torch.nn as nn
import onnx
import onnx_tf


class Model(nn.Module):
    def forward(self, x):
        x = nn.functional.upsample(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


model = Model()
dummy_input = torch.arange(1,5).view(1,1,2,2).float()
torch.onnx.export(model, dummy_input, "dummy.onnx")

onnx_model = onnx.load("dummy.onnx")
tf_model = onnx_tf.backend.prepare(onnx_model)

onnx

tensorflow

onnx_tf

torch.onnx

bilinear

linear

onnx

onnx-tensorflow

nearest

bilinear

upsample

onnx