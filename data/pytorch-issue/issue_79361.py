import torch.nn as nn

import torch
import numpy as np
from onnx.backend.test.case.node import resize
out_size = [1, 1]


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=out_size, mode='bilinear')


model = Model()
x = torch.rand([1, 1, 4, 4]).numpy()
y_torch = model(torch.from_numpy(x)).detach().numpy()
y_onnx_pytorch_hp = resize.interpolate_nd(
    x, resize.linear_coeffs, output_size=[1, 1, *out_size], coordinate_transformation_mode='pytorch_half_pixel')
y_onnx_hp = resize.interpolate_nd(
    x, resize.linear_coeffs, output_size=[1, 1, *out_size], coordinate_transformation_mode='half_pixel')

print('Comparing PyTorch to ONNX half_pixel')
np.testing.assert_allclose(
    y_torch, y_onnx_hp, err_msg='PyTorch vs half_pixel')  # succeed
print('Passed')
print('Comparing PyTorch to ONNX pytorch_half_pixel')
np.testing.assert_allclose(y_torch, y_onnx_pytorch_hp,
                           err_msg='PyTorch vs pytorch_half_pixel')  # fail
print('Passed')


torch.onnx.export(model, (torch.from_numpy(x), ),
                  "output.onnx", opset_version=14, input_names=["input"], output_names=["output"])
# output.onnx uses pytorch_half_pixel

import onnxruntime as ort
y_ort = ort.InferenceSession("output.onnx", provider_options=[
                             "CPUExecutionProvider"]).run(["output"], {"input": x})[0]

print('Comparing ORT pytorch_half_pixel to ORT pytorch_half_pixel')
np.testing.assert_allclose(y_ort, y_onnx_pytorch_hp,
                           err_msg='ORT vs pytorch_half_pixel')  # succeed
print('Passed')