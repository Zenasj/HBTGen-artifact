import torch.nn as nn

import torch

padding_3 = 3
input_3 = torch.randn(16, 3, 8, 320, 480, dtype=torch.float16)

m3d = torch.nn.ReplicationPad3d(padding=padding_3)
print(m3d(input_3))
# RuntimeError: "replication_pad3d" not implemented for 'Half'