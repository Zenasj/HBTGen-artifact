py
import torch
import torch.nn as nn

dtype = torch.quint8
weight = nn.Parameter(torch.ones(10, dtype=dtype))

# RuntimeError: false INTERNAL ASSERT FAILED at 
# "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/quantized/Quantizer.cpp":444,
# please report a bug to PyTorch. 
# cannot call qscheme on UnknownQuantizer