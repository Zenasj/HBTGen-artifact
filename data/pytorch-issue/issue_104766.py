import torch
from timm.models.vision_transformer import vit_tiny_patch16_224

model = vit_tiny_patch16_224()
model.to("cuda")
input = torch.rand([1,3,224,224],device="cuda")
output = model(input)
output.mean().backward()

import torch
from timm.models.vision_transformer import vit_tiny_patch16_224

model = vit_tiny_patch16_224()
for block in model.blocks:
     block.attn.fused_attn = False
model.to("cuda")
input = torch.rand([1,3,224,224],device="cuda")
output = model(input)
output.mean().backward()