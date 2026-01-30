#!/usr/bin/env python
import os
import torch
import onnxruntime
import numpy as np

def main():
    import io
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    INPUT_LEN = 12
    NUM_GESTURES = 6
    class aslTinyModel(nn.Module):
        def __init__(self):
            super(aslTinyModel, self).__init__()
            self.quant = torch.quantization.QuantStub()
            self.fc1 = nn.Linear(INPUT_LEN, 8)
            self.fc2 = nn.Linear(8, 4)
            self.fc3 = nn.Linear(4, NUM_GESTURES)
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = x.view((-1, INPUT_LEN))
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.dequant(h)
            return h
    model = aslTinyModel()
    ### TRAIN THE MODEL ###

    ###  QUANTIZE MODEL ###
    import torch.quantization
    num_calibration_batches = 10
    model.eval()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.quantization.default_qconfig
    qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    ### EXPORT TO ONNX ###
    dummy_input = torch.randn(INPUT_LEN)
    input_names = ["x"]
    outputs = model(dummy_input)

    traced = torch.jit.trace(model, dummy_input)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)

    model = torch.jit.load(buf)
    f = io.BytesIO()
    torch.onnx.export(model, dummy_input, f, input_names=input_names, example_outputs=outputs,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    f.seek(0)

if __name__ == "__main__":
    main()