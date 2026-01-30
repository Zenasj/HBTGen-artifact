import torch.nn as nn
import random

import torch


class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,  # doesn't affect errors
            window=self._window,
            pad_mode="constant",  # aten.reflection_pad1d unsupported op
        )
        return x


m = STFTModel()

# Shape [B, T] audio signals
input_signals = torch.randn([2, 16000])

args = (input_signals,)
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
torch.onnx.dynamo_export(
    m,
    *args,
    export_options=export_options,
)

import onnx
import torch
import numpy as np
import onnxruntime as ort


class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=self._window,
            pad_mode="constant",  # aten.reflection_pad1d unsupported op
        )
        return x


m = STFTModel()
m.eval()

batch_size = 2
signal_length = 16000

# Export
# Shape [B, T] audio signals
input_signals = torch.randn([batch_size, signal_length])
args = (input_signals,)
# Note: static dims
export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
exported_model = torch.onnx.dynamo_export(
    m,
    *args,
    export_options=export_options,
)
exported_model.save("tmp.onnx")

# Load and attempt to run
onnx_model = onnx.load("tmp.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX check ok")
print("Instantiate session")
session: ort.InferenceSession = ort.InferenceSession(
    "tmp.onnx", providers=["CPUExecutionProvider"]
)
# Need static shape... use same as exported
np_signals = np.random.random(size=[batch_size, signal_length]).astype(np.float32)
print(f"Run ONNX graph with signals of shape {np_signals.shape}")
# Exporter also gives parameter a weird name: signals -> l_signals_
outputs = session.run(None, {"l_signals_": np_signals})

import onnx
import torch
import numpy as np
import onnxruntime as ort

import onnx.inliner
import onnx.reference


class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=self._window,
            pad_mode="constant",  # aten.reflection_pad1d unsupported op
        )
        return x


m = STFTModel()
m.eval()

# NOTE: Change batch_size to 1, 2 to see different errors
batch_size = 1
signal_length = 16000

# Export
# Shape [B, T] audio signals
input_signals = torch.randn([signal_length])
args = (input_signals,)
# Note: static dims
export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
exported_model = torch.onnx.dynamo_export(
    m,
    *args,
    export_options=export_options,
)

print("output shape", m(input_signals).shape)

# Load and attempt to run
# NOTE: Start from here to load the model and reproduce error
onnx_model = exported_model.model_proto
print("ONNX check ok")
# onnx_model = onnx.inliner.inline_local_functions(onnx_model)
# onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True, data_prop=True)
onnx.save_model(onnx_model, f"stft_inlined_batch_{batch_size}.onnx")
print("Instantiate session")
session: ort.InferenceSession = ort.InferenceSession(
    onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
)
# Need static shape... use same as exported
np_signals = input_signals.numpy()
print(f"Run ONNX graph with signals of shape {np_signals.shape}")
# Exporter also gives parameter a weird name: signals -> l_signals_
outputs = session.run(None, {"arg0": np_signals})
expected = torch.view_as_real(m(input_signals)).numpy()

print(outputs)
print(expected)

np.testing.assert_allclose(outputs[0], expected)

# session = onnx.reference.ReferenceEvaluator(onnx_model, verbose=10)
# session.run(None, {"arg0": np_signals})