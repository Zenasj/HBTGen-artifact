import torch

#buf1193.size()=torch.Size([8, 48, 24, 16]) buf1196.size()=torch.Size([192, 48, 1, 1]) buf1193.stride()=(18432, 384, 16, 1) buf1196.stride()=(48, 1, 1, 1)
buf1197 = aten.convolution(buf1193, buf1196, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)