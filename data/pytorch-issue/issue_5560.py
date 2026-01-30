import torch
import torch.nn as nn

optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

prediction = model(batch_input)
loss = criterion(torch.sigmoid(prediction), label)

optimizer.zero_grad()
loss.backward()
optimizer.step()