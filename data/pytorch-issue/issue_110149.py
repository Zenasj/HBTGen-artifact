import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return x

model = Net(512, 1)
x = torch.randn((2, 100,  512))
model.eval()
with torch.no_grad():
    model(x)