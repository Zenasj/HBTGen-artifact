import torch

@torch.inference_mode()
def fa():
    print(f'a: inference mode {torch.is_inference_mode_enabled()}')
    fb()

@torch.inference_mode(False)
def fb():
    print(f'b: inference mode {torch.is_inference_mode_enabled()}')

fa()
fb()
with torch.inference_mode():
    fb()