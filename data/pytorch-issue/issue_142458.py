import torch.nn as nn

import torch

class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # Defining window within forward causes it to break
        window = torch.hann_window(window_length=320)
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=window, # using self._window avoids the issue
            pad_mode="constant",
        )

        return x


m = STFTModel()

# Shape [B, T] audio signals
input_signals = torch.randn([2, 16000]).cpu()

args = (input_signals,)
ep = torch.onnx.export( # note that torch.export.export works
    m,
    args,
    dynamo=True
)

print(ep)

# ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, c__window: "f32[320]", signals: "f32[2, 16000]"):
         # File: /home/justinchu/dev/onnxscript/testt.py:11 in forward, code: x = signals.stft(
        view: "f32[1, 2, 16000]" = torch.ops.aten.view.default(signals, [1, 2, 16000]);  signals = None
        constant_pad_nd: "f32[1, 2, 16512]" = torch.ops.aten.constant_pad_nd.default(view, [256, 256], 0.0);  view = None
        view_1: "f32[2, 16512]" = torch.ops.aten.view.default(constant_pad_nd, [2, 16512]);  constant_pad_nd = None
        constant_pad_nd_1: "f32[512]" = torch.ops.aten.constant_pad_nd.default(c__window, [96, 96]);  c__window = None
        unfold: "f32[2, 101, 512]" = torch.ops.aten.unfold.default(view_1, -1, 512, 160);  view_1 = None
        mul: "f32[2, 101, 512]" = torch.ops.aten.mul.Tensor(unfold, constant_pad_nd_1);  unfold = constant_pad_nd_1 = None
        _fft_r2c: "c64[2, 101, 257]" = torch.ops.aten._fft_r2c.default(mul, [2], 0, True);  mul = None
        permute: "c64[2, 257, 101]" = torch.ops.aten.permute.default(_fft_r2c, [0, 2, 1]);  _fft_r2c = None
        return (permute,)

# ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, signals: "f32[2, 16000]"):
         # File: /home/justinchu/dev/onnxscript/testt.py:10 in forward, code: window = torch.hann_window(window_length=320)
        hann_window: "f32[320]" = torch.ops.aten.hann_window.default(320, device = device(type='cpu'), pin_memory = False)
        
         # File: /home/justinchu/dev/onnxscript/testt.py:11 in forward, code: x = signals.stft(
        view: "f32[1, 2, 16000]" = torch.ops.aten.view.default(signals, [1, 2, 16000]);  signals = None
        constant_pad_nd: "f32[1, 2, 16512]" = torch.ops.aten.constant_pad_nd.default(view, [256, 256], 0.0);  view = None
        view_1: "f32[2, 16512]" = torch.ops.aten.view.default(constant_pad_nd, [2, 16512]);  constant_pad_nd = None
        constant_pad_nd_1: "f32[512]" = torch.ops.aten.constant_pad_nd.default(hann_window, [96, 96]);  hann_window = None
        unfold: "f32[2, 101, 512]" = torch.ops.aten.unfold.default(view_1, -1, 512, 160);  view_1 = None
        mul: "f32[2, 101, 512]" = torch.ops.aten.mul.Tensor(unfold, constant_pad_nd_1);  unfold = constant_pad_nd_1 = None
        _fft_r2c: "c64[2, 101, 257]" = torch.ops.aten._fft_r2c.default(mul, [2], 0, True);  mul = None
        permute: "c64[2, 257, 101]" = torch.ops.aten.permute.default(_fft_r2c, [0, 2, 1]);  _fft_r2c = None
        return (permute,)

import torch

class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # Defining window within forward causes it to break
        window = torch.hann_window(window_length=320)
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=window, # using self._window avoids the issue
            pad_mode="constant",
        )

        return x


m = STFTModel()

# Shape [B, T] audio signals
input_signals = torch.randn([2, 16000]).cpu()

args = (input_signals,)
ep = torch.export.export( # note that torch.export.export works
    m,
    args,
    strict=False
)
ep = ep.run_decompositions()
print(ep)