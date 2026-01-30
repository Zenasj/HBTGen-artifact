import torch
bf16 = torch.cuda.is_bf16_supported()
print(f'{bf16=}')