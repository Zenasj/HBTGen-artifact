import torch

("scatter_add", (torch.zeros(2, 2, 2, dtype=torch.float16, device=dev, names=('N', 'C', 'L')),
                             'C',
                             torch.randint(0, 2, (2, 2, 2), device=dev),
                             torch.randn((2, 2, 2), dtype=torch.float32, device=dev))),