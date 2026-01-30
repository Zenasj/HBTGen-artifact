import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda"
dtype = torch.float64
torch.set_default_dtype(torch.float64)

X = torch.randn(10, 1, device=device, dtype=dtype)
y = 2 * X + 1 + 0.1 * torch.randn(10, 1, device=device, dtype=dtype)  

class LinearRegression(nn.Module):
    def __init__(self, device: str = None):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1, device=device)
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()