import torch.nn as nn

from tempfile import TemporaryFile
import torch  # v2.5.1

with TemporaryFile() as file:
    model = torch.nn.Linear(3, 3)
    exported_model = torch.export.export(model, args=(torch.randn(3),))
    # "BufferedRandom" cannot be assigned to type "str | PathLike[Unknown] | BytesIO"
    torch.export.save(exported_model, file)  # ❌ typing error!