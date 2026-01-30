import torch

device = 'cuda:0'
num_channel = 2 ** 6

for dtype in [torch.float64, torch.float32, torch.float16]:
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = 2 ** 31 // element_size

    t = torch.zeros([num_channel, numel // num_channel], dtype=dtype, device=device)
    t[num_channel // 2] = 1

    print(f"{dtype}, small tensor")
    print("Indices from torch.argmax():", torch.argmax(t, dim=0))
    print("Indices from torch.max():", torch.max(t, dim=0)[1])
    print()

    t = torch.zeros([num_channel, numel // num_channel + 1], dtype=dtype, device=device)
    t[num_channel // 2] = 1

    print(f"{dtype}, big tensor")
    print("Indices from torch.argmax():", torch.argmax(t, dim=0))
    print("Indices from torch.max():", torch.max(t, dim=0)[1])
    print()