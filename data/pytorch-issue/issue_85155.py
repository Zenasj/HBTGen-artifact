import torch.nn as nn

import torch
layer = torch.nn.PixelShuffle(1)

model_input = torch.ones((1,1,1,1,0))
pred = layer(model_input)

model_input = torch.ones((1,1,1,1,0))
model_input = torch.ones((1,1,1,0,1))
model_input = torch.ones((1,1,0,1,1))

py
TORCH_CHECK(self.size(-1) > 0, error_message, upscale_factor);
TORCH_CHECK(self.size(-2) > 0, error_message, upscale_factor);
TORCH_CHECK(self.size(-3) > 0, error_message, upscale_factor);