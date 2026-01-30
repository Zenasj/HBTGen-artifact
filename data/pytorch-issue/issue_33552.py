import torch
import torch.nn as nn

device = 'cuda'
model = nn.LSTM(10, 16, bidirectional = False, batch_first=True)
model.to(device)

seq_len = 5
batch_size = 8
input_dim = 10
inp = torch.randn(batch_size, seq_len, input_dim).to(device)

# Switch
if True:
    model = nn.DataParallel(model)

output = model(inp)[0]
output.mean().backward()
print([p.grad for p in model.parameters()])