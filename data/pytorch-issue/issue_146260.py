import torch

_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")