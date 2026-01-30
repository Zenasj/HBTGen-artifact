import torch.nn as nn

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
            pad_mode="constant",
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
exported_program = torch.export.export(m, args)
print(exported_program)
exported_model = torch.onnx.dynamo_export(
    # m,
    exported_program,
    *args,
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

np.testing.assert_allclose(outputs[0], expected)

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
            pad_mode="constant",
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
exported_program = torch.export.export(m, args)
print(exported_program)
exported_model = torch.onnx.dynamo_export(
    # m,
    exported_program,
    *args,
)

outputs = exported_model(*args)
expected = torch.view_as_real(m(input_signals)).numpy()

np.testing.assert_allclose(outputs[0], expected, atol=1e-4, rtol=1e-4)