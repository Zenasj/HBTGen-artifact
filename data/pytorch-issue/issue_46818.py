# torch.rand(1, 172, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_features, num_hiddens, num_outputs):
        super(MyModel, self).__init__()
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        
        # Define all layers in the __init__ method
        self.linear1 = nn.Linear(num_features, num_hiddens)
        self.bn1 = nn.BatchNorm1d(num_hiddens)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.15)
        
        self.linear2 = nn.Linear(num_hiddens, int(0.9 * num_hiddens))
        self.bn2 = nn.BatchNorm1d(int(0.9 * num_hiddens))
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.15)
        
        self.linear3 = nn.Linear(int(0.9 * num_hiddens), int(0.9 * 0.9 * num_hiddens))
        self.bn3 = nn.BatchNorm1d(int(0.9 * 0.9 * num_hiddens))
        self.tanh3 = nn.Tanh()
        self.dropout3 = nn.Dropout(p=0.15)
        
        self.linear4 = nn.Linear(int(0.9 * 0.9 * num_hiddens), int(0.9 * 0.9 * 0.9 * num_hiddens))
        self.bn4 = nn.BatchNorm1d(int(0.9 * 0.9 * 0.9 * num_hiddens))
        self.tanh4 = nn.Tanh()
        self.dropout4 = nn.Dropout(p=0.15)
        
        self.linear5 = nn.Linear(int(0.9 * 0.9 * 0.9 * num_hiddens), num_outputs)
    
    def forward(self, X):
        X = self.linear1(X)
        X = self.bn1(X)
        X = self.tanh1(X)
        X = self.dropout1(X)
        
        X = self.linear2(X)
        X = self.bn2(X)
        X = self.tanh2(X)
        X = self.dropout2(X)
        
        X = self.linear3(X)
        X = self.bn3(X)
        X = self.tanh3(X)
        X = self.dropout3(X)
        
        X = self.linear4(X)
        X = self.bn4(X)
        X = self.tanh4(X)
        X = self.dropout4(X)
        
        X = self.linear5(X)
        
        return X

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(172, 1150, 3)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 172))

