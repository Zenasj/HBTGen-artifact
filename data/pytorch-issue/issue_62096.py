import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=32, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(32, 2)
    
    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = torch.mean(outputs, 1)
        outputs = self.linear(outputs)
        return outputs

class Data(Dataset):
    def __init__(self):
        super().__init__()
        self.data = np.random.rand(300).astype(np.float32).reshape(10, 5, 6)
        self.labels = np.random.choice(2, 10).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(model, dataloader, criterion, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def run_experiment(epochs=2):
    
    train_dataset = Data()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NN().to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        train(model, train_loader, criterion, optimizer)    
    
    return copy.deepcopy(model)

def compare_models(model_1, model_2):
    models_differ = 0
    for kv_1, kv_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        assert kv_1[0] == kv_2[0]
        if not torch.equal(kv_1[1], kv_2[1]):
            models_differ += 1
            print('Mismtach found: ', kv_1[0])
    if models_differ == 0:
        print('Models match.')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    SEED = 42    
    
    set_seed(SEED)
    m1 = run_experiment()
    
    set_seed(SEED)
    m2 = run_experiment()
    
    compare_models(m1, m2)