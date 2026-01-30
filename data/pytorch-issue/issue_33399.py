import torch.nn as nn

import torch.onnx; import torch.nn; torch.onnx.export(torch.nn.CELU(alpha=2), torch.rand(1), 'tst.onnx')