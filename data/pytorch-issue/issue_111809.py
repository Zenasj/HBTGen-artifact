import open_clip
import torch
model = open_clip.create_model('xlm-roberta-base-ViT-B-32', force_custom_text=True, pretrained_hf=False).text
example_input = torch.ones((1, 514), dtype=torch.int64)
model(example_input)

import open_clip
import torch
from torch.utils.flop_counter import FlopCounterMode

model = open_clip.create_model('xlm-roberta-base-ViT-B-32', force_custom_text=True, pretrained_hf=False).text
example_input = torch.ones((1, 514), dtype=torch.int64)
flop_counter = FlopCounterMode(model)
with flop_counter:
    model(example_input)