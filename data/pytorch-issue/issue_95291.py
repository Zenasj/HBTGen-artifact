py
import torch

torch.manual_seed(420)

x = torch.rand(1)
y = torch.quantize_per_tensor(x, scale=0.5, zero_point=0, dtype=torch.qint8)
torch.equal(x, y)

# RuntimeError: self.is_quantized() INTERNAL ASSERT FAILED at 
# "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/quantized/Quantizer.cpp":82, 
# please report a bug to PyTorch. get_qtensorimpl: not a quantized tenso