import torch
print('device dtype arange-floor-result')
for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.float64]:
        result = torch.arange(-5, 5, 1.4, device=device, dtype=dtype)
        print(f'{device} {dtype} {result.floor()}')
print()
print('device dtype arange-item-5')
for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.float64]:
        result = torch.arange(-5, 5, 1.4, device=device, dtype=dtype)[5].item()
        print(f'{device} {dtype} {result}')