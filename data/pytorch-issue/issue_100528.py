import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
net = nn.Sequential(nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
), nn.Sequential(
    nn.Linear(1, 20),
    nn.ReLU(),
    nn.Linear(20, 1)))

# Create dummy data
train_data = torch.randn(100, 10, requires_grad=True)
train_labels = torch.randn(100, 1)

# Define the loss function and optimizer
criterion = nn.MSELoss()

# Train the neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def _pre_hook(module, *_, **__):
    print('Push: ', f'{module.__class__.__name__}')
def _after_hook(module, *_, **__):
    print('Pop: ', f'{module.__class__.__name__}')
torch.nn.modules.module.register_module_full_backward_pre_hook(_pre_hook)
torch.nn.modules.module.register_module_full_backward_hook(_after_hook)

print('------ train_data.requires_grad = True ------')

for epoch in range(1):
    running_loss = 0.0
    for i in range(1):
        inputs, labels = train_data[i].to(device), train_labels[i].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / 100))
    running_loss = 0.0

print('------ train_data.requires_grad = False ------')

train_data.requires_grad = False
for epoch in range(1):
    running_loss = 0.0
    for i in range(1):
        inputs, labels = train_data[i].to(device), train_labels[i].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / 100))
    running_loss = 0.0

print('Finished Training')