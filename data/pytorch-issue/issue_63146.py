import torch
from repro63146.modeling import VisionTransformer, CONFIGS

config = CONFIGS['ViT-B_16']
model = VisionTransformer(config, 224, zero_head=True, num_classes=10)
x = torch.randn(1, 512, 10, 10)
dummy_input = torch.randn(64, 3, 224, 224)
model(dummy_input)