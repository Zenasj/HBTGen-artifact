import torch
import torch.nn as nn

model = nn.EmbeddingBag(10, 3, mode='sum')
input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
offsets = torch.LongTensor([0, 4])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

result = model(input, offsets)
print(result)

input, offsets = input.unsqueeze(0).expand(torch.cuda.device_count(), input.size(0)), offsets.unsqueeze(0).expand(torch.cuda.device_count(), offsets.size(0))

input, offsets = input.squeeze(0), offsets.squeeze(0)