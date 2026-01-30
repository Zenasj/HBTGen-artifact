import torch.nn as nn

import copy
import torch
import torch
from torch import nn
import numpy as np
import os, sys
import onnx
import onnxruntime

y = torch.randn(1, 2, 3, 6)
device = 'cuda'
ygpu = y.to(device=device)
#conv = nn.Conv2d(in_channels=57, out_channels=1, kernel_size=(1,3), stride=2)
gpuconv = nn.Sequential(
nn.Conv2d(2, 2, stride=1, kernel_size=3),
nn.BatchNorm2d(num_features=2)
).to(device=device)
xgpu = gpuconv(ygpu)
gpuconv.eval()
xgpu = gpuconv(ygpu)
conv = copy.deepcopy(gpuconv).to(device='cpu')
x = conv(y)

print(torch.allclose(x, xgpu.to(device='cpu'), rtol=1e-03, atol=1e-05))
torch.onnx.export(conv, y, 'test.onnx', export_params=True)
torch.onnx.export(gpuconv, ygpu, 'testgpu.onnx', export_params=True)
onnx_model = onnx.load('testgpu.onnx')
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession('testgpu.onnx')

def to_numpy(tensor):
    return np.array(tensor.to('cpu').detach().numpy() if tensor.requires_grad else tensor.cpu().numpy())

x = torch.tensor([[[[ 1.1227, -0.5545, -0.7871,  0.0841,  0.3714,  0.1612],
          [-0.2052, -0.6623,  1.6033,  2.1245,  1.9609, -1.6847],
          [ 0.1736, -0.7240, -0.0737,  0.3207,  1.5287, -1.3479]],

         [[-0.0267,  1.3100, -0.5756,  0.9288,  1.0432, -0.4871],
          [-0.9110, -0.2054, -1.6963,  0.5995,  1.7877, -2.0397],
          [-0.1205, -1.0103, -2.1805,  1.4364, -1.3835,  0.6537]]]])

torch.manual_seed(0)

device = "mps"
x_m1 = x.to(device=device)
conv_m1 = nn.Sequential(
    nn.Conv2d(2, 2, stride=1, kernel_size=3),
    nn.BatchNorm2d(num_features=2)
).to(device=device)

y_m1 = conv_m1(x_m1)
conv_m1.eval()
eval_m1 = conv_m1(x_m1)

conv = copy.deepcopy(conv_m1).to(device='cpu')
y = conv(x)

print(torch.allclose(y, y_m1.to(device='cpu'), rtol=1e-03, atol=1e-05))
torch.onnx.export(conv, x, 'test.onnx', export_params=True)
torch.onnx.export(conv_m1, x_m1, 'testm1.onnx', export_params=True)
onnx_model = onnx.load('testm1.onnx')
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession('testm1.onnx')

import copy

import numpy as np
import onnxruntime
import torch
from torch import nn

y = torch.randn(1, 2, 3, 6)
device = torch.device("mps")
y_m1 = y.to(device=device)
m1conv = nn.Sequential(
    nn.Conv2d(2, 2, stride=1, kernel_size=3), nn.BatchNorm2d(num_features=2)
).to(device=device)
x_m1 = m1conv(y_m1)
m1conv.eval()
x_m1 = m1conv(y_m1)
conv = copy.deepcopy(m1conv).to(device="cpu")
x = conv(y)

print(torch.allclose(x, x_m1.to(device="cpu"), rtol=1e-03, atol=1e-05))
torch.onnx.export(conv, y, "test.onnx", export_params=True)
torch.onnx.export(m1conv, y_m1, "testm1.onnx", export_params=True)

ort_session = onnxruntime.InferenceSession("test.onnx")
ort_cpu = ort_session.run(None, {"input.1": y_m1.cpu().numpy()})
print("cpu", ort_cpu)

ort_session = onnxruntime.InferenceSession("testm1.onnx")
ort_m1 = ort_session.run(None, {"input.1": y_m1.cpu().numpy()})
print("m1", ort_m1)

print("pytorch_cpu", x)
print("pytorch_m1", x_m1)

print("assertion...")
torch.testing.assert_allclose(ort_m1[0], x_m1.to(device="cpu"), rtol=1e-03, atol=1e-05)
torch.testing.assert_allclose(ort_cpu[0], x_m1.to(device="cpu"), rtol=1e-03, atol=1e-05)
print("passed")

import copy
import torch
from torch import nn
import onnx
import onnxruntime
import numpy as np
y = torch.randn(1, 2, 3, 6)
device = 'mps'
ygpu = y.to(device=device)

gpuconv = nn.Sequential(
            nn.Conv2d(2, 2, stride=1, kernel_size=3),
            nn.BatchNorm2d(num_features=2)
            ).to(device=device)
xgpu = gpuconv(ygpu)
gpuconv.eval()
xgpu = gpuconv(ygpu)
conv = copy.deepcopy(gpuconv).to(device='cpu')
x = conv(y)

print(torch.allclose(x, xgpu.to(device='cpu'),  rtol=1e-03, atol=1e-05))
torch.onnx.export(conv, y, 'test.onnx', export_params=True)
torch.onnx.export(gpuconv, ygpu, 'testgpu.onnx', export_params=True)
onnx_model = onnx.load('testgpu.onnx')
onnx.checker.check_model(onnx_model)



def to_numpy(tensor):
    return np.array(tensor.to('cpu').detach().numpy() if tensor.requires_grad else tensor.cpu().numpy())

# compute ONNX Runtime output prediction using GPU version
ort_session = onnxruntime.InferenceSession('testgpu.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(y)}
ort_outs = np.array(ort_session.run(None, ort_inputs)).squeeze()
torch_out_numpy = to_numpy(x).squeeze()
print(np.allclose(torch_out_numpy,ort_outs, rtol=1e-03, atol=1e-05))

# compute ONNX Runtime output prediction using CPU version
ort_session = onnxruntime.InferenceSession('test.onnx')
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(y)}
ort_outs = np.array(ort_session.run(None, ort_inputs)).squeeze()
torch_out_numpy = to_numpy(x).squeeze()
print(np.allclose(torch_out_numpy,ort_outs, rtol=1e-03, atol=1e-05))

import copy
import os
import tempfile

import onnxruntime

import torch
from torch import nn


def reveal_error():
    torch.manual_seed(0)
    y = torch.randn(1, 2, 3, 6)
    mps = torch.device("mps")
    cpu = torch.device("cpu")
    y_m1 = y.to(device=mps)
    m1conv = nn.Sequential(
        nn.Conv2d(2, 2, stride=1, kernel_size=3), nn.BatchNorm2d(num_features=2)
    ).to(device=mps)
    x_m1 = m1conv(y_m1)
    m1conv.eval()
    x_m1 = m1conv(y_m1)
    conv = copy.deepcopy(m1conv).to(device=cpu)
    x = conv(y)

    print(torch.allclose(x, x_m1.to(device=cpu), rtol=1e-03, atol=1e-05))

    with tempfile.TemporaryDirectory() as tempdir:
        cpu_model_path = os.path.join(tempdir, "test.onnx")
        m1_model_path = os.path.join(tempdir, "testm1.onnx")

        torch.onnx.export(conv, y, cpu_model_path, export_params=True)
        torch.onnx.export(m1conv, y_m1, m1_model_path, export_params=True)

        print("models saved to", tempdir)

        ort_session = onnxruntime.InferenceSession(cpu_model_path)
        ort_cpu = ort_session.run(None, {"input.1": y_m1.cpu().numpy()})
        print("cpu", ort_cpu)

        ort_session = onnxruntime.InferenceSession(m1_model_path)
        ort_m1 = ort_session.run(None, {"input.1": y_m1.cpu().numpy()})
        print("m1", ort_m1)

        print("pytorch_cpu", x)
        print("pytorch_m1", x_m1)

        print("assertion...")
        torch.testing.assert_allclose(
            ort_m1[0], x_m1.to(device=cpu), rtol=1e-03, atol=1e-05
        )
        torch.testing.assert_allclose(
            ort_cpu[0], x_m1.to(device=cpu), rtol=1e-03, atol=1e-05
        )
        print("passed")


def main():
    reveal_error()


if __name__ == "__main__":
    main()