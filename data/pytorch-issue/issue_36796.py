import torch.nn as nn

import torch
import onnxruntime

bug = True

class Preemphasis(torch.nn.Module):
    def forward(self, signal):
        if bug:
            signal = signal.squeeze(1)
        signal = torch.cat([signal[..., :1], signal[..., 1:] - 0.97 * signal[..., :-1]], dim = -1)
        return signal

frontend = Preemphasis()
input = torch.rand(16, 8000)
torch.onnx.export(frontend, (input,), 'model.onnx', opset_version = 10, export_params = True, do_constant_folding = True, input_names = ['signal'], output_names = ['output'], dynamic_axes = dict(signal = {0 : 'B', 1 : 'T'}, output = {0 : 'B', 1: 'T'}))
(output_, ) = onnxrt_session.run(None, dict(signal = input.cpu().numpy()))

output = frontend(input)
assert torch.allclose(output.cpu(), torch.from_numpy(output_), rtol = 1e-02, atol = 1e-03)

def squeeze(x, dim):
   return x.squeeze(dim) if x.size(dim) == 1 else x