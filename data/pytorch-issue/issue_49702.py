import torch
import torch.nn as nn

[mypy-torch.nn.qat.modules.activations]
ignore_errors = True

[mypy-torch.nn.qat.modules.conv]
ignore_errors = True

[mypy-torch.nn.quantized.dynamic.modules.linear]
ignore_errors = True