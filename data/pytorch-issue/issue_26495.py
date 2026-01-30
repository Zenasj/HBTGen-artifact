import torch
empty = torch.randn([0])
tensor = torch.randn(2, 3)

result = torch.pow(tensor, 0, out=empty.clone())