import torch.nn as nn

#!/usr/bin/env python3
import torch
model = torch.nn.Conv2d(8, 4, 3).cuda().half()
model = model.to(memory_format=torch.channels_last)
input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, requires_grad=True)
input = input.to(device="cuda", memory_format=torch.channels_last, dtype=torch.float16)

# should print True for is_contiguous(channels_last), and strides must match NHWC format
print(input.is_contiguous(memory_format=torch.channels_last), input.shape, input.stride() )

out = model(input)

# should print True for is_contiguous(channels_last), and strides must match NHWC format
print("Contiguous channel last :", out.is_contiguous(memory_format=torch.channels_last), " out shape :",  out.shape, "out stride :", out.stride() )