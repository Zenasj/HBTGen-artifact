import torch
import torch.nn as nn
input_data = torch.randn(1, 10)
model = nn.Linear(10, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, capturable=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, capturable=True)
optimizer.zero_grad()
criterion = nn.MSELoss()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, input_data)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{(epoch + 1)}/{100}], Loss: {loss.item():.4f}')

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a graph from device cuda/xpu/hpu/privateuseone/xla. 
            Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)"""