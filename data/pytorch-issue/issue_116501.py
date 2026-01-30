import torch.nn as nn

import torch
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

a = torch.rand(2, 3, 224, 224, dtype=torch.float32).cuda()
b = torch.rand(2, 3, 224, 224, dtype=torch.float32).cuda()
c = torch.cat([a, b])
conv = torch.nn.Conv2d(in_channels=3, out_channels=1408, kernel_size=(14,14), stride=(14,14)).to(dtype=torch.float32).cuda()
output0 = conv(a)
output1 = conv(b)
output01 = conv(c)
print(torch.allclose(a, c[:2]))
print(torch.allclose(b, c[2:]))
print(torch.allclose(output0, output01[:2]))
print(torch.allclose(output1, output01[2:]))
# output:
#     True
#     True
#     False
#     False

a = torch.rand(2, 3, 224, 224, dtype=torch.bfloat16).cuda()
b = torch.rand(2, 3, 224, 224, dtype=torch.bfloat16).cuda()
c = torch.cat([a, b])
conv = torch.nn.Conv2d(in_channels=3, out_channels=1408, kernel_size=(14,14), stride=(14,14)).to(dtype=torch.bfloat16).cuda()
output0 = conv(a)
output1 = conv(b)
output01 = conv(c)
print(torch.allclose(a, c[:2]))
print(torch.allclose(b, c[2:]))
print(torch.allclose(output0, output01[:2]))
print(torch.allclose(output1, output01[2:]))
# output:
#     True
#     True
#     True
#     True