import torch.nn as nn
import random

import io

import numpy as np
import torch


class QAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant1 = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        res = torch.nn.functional.avg_pool2d(
            self.quant1(x), kernel_size=2, stride=1, padding=0
        )
        return self.dequant(res)


def generic_test(
    model, sample_inputs, input_names=None, decimal=3, relaxed_check=False
):
    torch.backends.quantized.engine = "qnnpack"
    pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
    model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    q_model = torch.ao.quantization.prepare(model, inplace=False)
    q_model = torch.ao.quantization.convert(q_model, inplace=False)

    traced_model = torch.jit.trace(q_model, pt_inputs)
    buf = io.BytesIO()
    torch.jit.save(traced_model, buf)
    buf.seek(0)
    q_model = torch.jit.load(buf)

    q_model.eval()
    output = q_model(*pt_inputs)

    f = io.BytesIO()
    torch.onnx.export(
        q_model,
        pt_inputs,
        f,
        verbose=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        # Caffe2 doesn't support newer opset versions
        opset_version=9,
    )


def export_to_onnx(model, input, input_names):
    traced = torch.jit.trace(model, input)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)

    model = torch.jit.load(buf)
    f = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        f,
        verbose=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        # Caffe2 doesn't support newer opset versions
        opset_version=9,
    )


# 1

x = np.random.rand(1, 2, 8, 8).astype("float32")
generic_test(QAvgPool2dModule(), (x,), input_names=["x"], decimal=5)


# 2

class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.default_qconfig
        self.fc1 = torch.ao.quantization.QuantWrapper(
            torch.nn.Conv2d(3, 5, 2, bias=True).to(dtype=torch.float)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


torch.backends.quantized.engine = "qnnpack"
qconfig = torch.ao.quantization.default_qconfig
model = ConvModel()
model.qconfig = qconfig
model = torch.ao.quantization.prepare(model)
model = torch.ao.quantization.convert(model)

x_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
x = torch.from_numpy(x_numpy).to(dtype=torch.float)
outputs = model(x)
input_names = ["x"]
export_to_onnx(model, x, input_names)


# 3


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.default_qconfig
        self.fc1 = torch.ao.quantization.QuantWrapper(
            torch.nn.Linear(5, 10).to(dtype=torch.float)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


torch.backends.quantized.engine = "qnnpack"
qconfig = torch.ao.quantization.default_qconfig
model = LinearModel()
model.qconfig = qconfig
model = torch.ao.quantization.prepare(model)
model = torch.ao.quantization.convert(model)

x_numpy = np.random.rand(1, 2, 5).astype(np.float32)
x = torch.from_numpy(x_numpy).to(dtype=torch.float)
outputs = model(x)
input_names = ["x"]
export_to_onnx(model, x, input_names)