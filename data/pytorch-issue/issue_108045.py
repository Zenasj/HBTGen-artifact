import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

# device = torch.device("cpu") # cpu OK
device = torch.device("mps")  # mps NOT OK
X = torch.tensor([[[0.5,0.4], [0,0]],[[0.3,0.2], [0,0]],[[0.5,0.2], [0,0]],[[0.2,0.2], [0,0]]], dtype=torch.float32).to(device)
y = torch.tensor([1,0,0,0], dtype=torch.float32).to(device) # value of batch_labels depends only on y[0] when mps selected

print(X.shape)
print(y.shape)
print(y)
dataset = TensorDataset(X, y)
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

for i, (batch_data, batch_labels) in enumerate(train_loader):
    print(batch_data)
    print(batch_labels)
    break