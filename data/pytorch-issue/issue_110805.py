generator = torch.Generator(device).manual_seed(2034)

for i in range(N):
    image = model()

import torch

device = "cuda"

# operands shape: [torch.Size([1, 128, 1024, 1024]), torch.Size([1, 128, 1024, 1024])]

left = torch.load("./tensor_operands_0.pt")
right = torch.load("./tensor_operands_1.pt")

output = torch.add(left, right)

print(torch.isinf(output).sum())

tensor(58, device='cuda:0')

torch.tensor([-51296], dtype=torch.float16) + torch.tensor([-15040], dtype=torch.float16)
# tensor([-inf], dtype=torch.float16)