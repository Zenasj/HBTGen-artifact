import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

LR = 10


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.linear = nn.Linear(512, 32)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)


# Initialise model
model = ExampleModel()
model = model.cuda()

# Prune weights
lstm_weights = [
    p for p in dict(model.lstm.named_parameters()).keys() if "weight" in p
]
for p in lstm_weights:
    prune.l1_unstructured(model.lstm, p, 0.5)
    prune.remove(model.lstm, p)
model.lstm.flatten_parameters()  # Make parameters contiguous

criterion = nn.MSELoss()
for step in range(10):
    print(f"Step {step}")
    # Dummy training data
    x = torch.randn(1, 10, 512, device="cuda:0")
    y = torch.randn(1, 10, 32, device="cuda:0")

    # Forward pass
    model.zero_grad()
    z = model(x)

    # Compute loss & gradients
    loss = criterion(z, y)
    loss.backward()

    # Update params
    for p in model.parameters():
        p.data.add_(-LR, p.grad.data)

print("Done training")