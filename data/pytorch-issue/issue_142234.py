import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = SimpleModel(10, 20).to(device)
model2 = SimpleModel(20, 10).to(device)

optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
scaler = torch.GradScaler('cuda')

# Dummy data
inputs = torch.randn(32, 10).to(device)
targets = torch.randint(0, 10, (32,)).to(device)

model1.train()
model2.train()

with torch.autocast(device_type=device, enabled=True):  # Enable or disable autocast
    embeddings = model1(inputs)
    
    with torch.no_grad():
        scores = model2(embeddings)
    
    outputs = model2(embeddings)
    loss = nn.CrossEntropyLoss()(outputs, targets)  

optimizer.zero_grad()
scaler.scale(loss).backward()

# Check if gradients are None for model2
for name, param in model2.named_parameters():
    print(f"{name} grad is None: {param.grad is None}")