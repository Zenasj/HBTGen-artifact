import torch.nn as nn

import torch
import copy
import io
from torch.quantization import default_qconfig, quantize_dynamic

class LSTMDynamicModel(torch.nn.Module):
    def __init__(self):
        super(LSTMDynamicModel, self).__init__()
        self.qconfig = default_qconfig
        self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x):
        x = self.lstm(x)
        return x

d_in, d_hid = 2, 2
model = LSTMDynamicModel().eval()
cell = model.lstm

# Replace parameter values s.t. the range of values is exactly
# 255, thus we will have 0 quantization error in the quantized
# GEMM call. This i s for testing purposes.
#
# Note that the current implementation does not support
# accumulation values outside of the range representable by a
# 16 bit integer, instead resulting in a saturated value. We
# must take care that in our test we do not end up with a dot
# product that overflows the int16 range, e.g.
# (255*127+255*127) = 64770. So, we hardcode the test values
# here and ensure a mix of signedness.
vals = [[100, -155],
        [100, -155],
        [-155, 100],
        [-155, 100],
        [100, -155],
        [-155, 100],
        [-155, 100],
        [100, -155]]
if isinstance(cell, torch.nn.LSTM):
    num_chunks = 4
vals = vals[:d_hid * num_chunks]
cell.weight_ih_l0 = torch.nn.Parameter(
    torch.tensor(vals, dtype=torch.float),
    requires_grad=False)
cell.weight_hh_l0 = torch.nn.Parameter(
    torch.tensor(vals, dtype=torch.float),
    requires_grad=False)

ref = copy.deepcopy(cell)

model_int8 = quantize_dynamic(model=model, dtype=torch.qint8)
model_fp16 = quantize_dynamic(model=model, dtype=torch.float16)

# Smoke test extra reprs
cell_int8 = model_int8.lstm
cell_fp16 = model_fp16.lstm

assert type(cell_int8) == torch.nn.quantized.dynamic.LSTM, \
    'torch.nn.LSTM should be converted to torch.nn.quantized.dynamic.LSTM after quantize_dynamic'
assert type(cell_fp16) == torch.nn.quantized.dynamic.LSTM, \
    'torch.nn.LSTM should be converted to torch.nn.quantized.dynamic.LSTM after quantize_dynamic'

niter = 10
x = torch.tensor([[100, -155],
                  [-155, 100],
                  [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)

h0_vals = [[-155, 100],
           [-155, 155],
           [100, -155]]

hx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)
cx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)

if isinstance(ref, torch.nn.LSTM):
    hiddens = (hx, cx)

ref_out, ref_hid = ref(x, hiddens)

# Compare int8 quantized to unquantized
output_int8, final_hiddens_int8 = cell_int8(x, hiddens)

torch.testing.assert_allclose(output_int8, ref_out)

for out_val, ref_val in zip(final_hiddens_int8, ref_hid):
    torch.testing.assert_allclose(out_val, ref_val)

class ScriptWrapper(torch.nn.Module):
    def __init__(self, cell):
        super(ScriptWrapper, self).__init__()
        self.cell = cell

    def forward(self, x, hiddens):
        # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        return self.cell(x, hiddens)

# TODO: TorchScript overloads don't work without this wrapper
cell_script = torch.jit.trace(ScriptWrapper(cell_int8), (x, (hx, cx)))
# cell_script = torch.jit.script(ScriptWrapper(cell_int8))
out_script, hid_script = cell_script(x, hiddens)
for out_val, ref_val in zip(out_script, ref_out):
    torch.testing.assert_allclose(out_val, ref_val)

# Test save/load
b = io.BytesIO()
# torch.jit.save(cell_script, b)
torch.jit.save(cell_script, 'foo.zip')

loaded = torch.jit.load('foo.zip')
loaded(x, (hx, cx))