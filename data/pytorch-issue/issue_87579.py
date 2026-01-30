import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running on", device)

data = torch.zeros((5, 3, 10), device=device)

lstm = nn.LSTM(10, 20, 1, batch_first=True).to(device)
h0 = torch.zeros((1, 5, 20), device=device)
c0 = torch.zeros((1, 5, 20), device=device)

expected = torch.randn((5, 3, 20), device=device)

output, _ = lstm(data, (h0, c0))
output = output.sum()

# crash occurs here
output.backward()

import torch
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running on", device)

data = torch.zeros((5, 3, 10), device=device)

lstm = nn.LSTM(10, 20, 1, batch_first=True).to(device)
h0 = torch.zeros((1, 5, 20), device=device)
c0 = torch.zeros((1, 5, 20), device=device)

expected = torch.randn((5, 3, 20), device=device)

output, _ = lstm(data, (h0, c0))
output = output.sum()

# crash occurs here
output.backward()

# Output:
# Running on mps