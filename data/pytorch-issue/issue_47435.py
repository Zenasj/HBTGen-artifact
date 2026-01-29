# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential()
        in_channels = 1
        for i in range(21):
            self.cnn.add_module(f'conv_{i}', nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
            self.cnn.add_module(f'relu_{i}', nn.ReLU())
            in_channels = 32  # Maintain 32 channels after first layer
        
        # RNN submodule structure matching the issue's description
        self.rnn = nn.ModuleList()
        # Submodule 0 of RNN
        rnn0 = nn.ModuleDict({
            'rnn': nn.LSTM(32 * 28 * 28, 128, batch_first=True),
            'embedding': nn.Embedding(100, 32)  # Placeholder for compatibility
        })
        # Submodule 1 of RNN
        rnn1 = nn.ModuleDict({
            'rnn': nn.LSTM(128, 64, batch_first=True),
            'embedding': nn.Embedding(100, 128)  # Placeholder for compatibility
        })
        self.rnn.append(rnn0)
        self.rnn.append(rnn1)
    
    def forward(self, x):
        # Process through CNN
        x = self.cnn(x)
        # Flatten CNN output to match RNN input
        x = x.view(x.size(0), -1)  # (batch, 32*28*28)
        x = x.unsqueeze(1)  # Add sequence dimension (batch, 1, features)
        
        # Process through RNN submodules
        x, _ = self.rnn[0]['rnn'](x)  # Use first RNN's LSTM
        x, _ = self.rnn[1]['rnn'](x)  # Use second RNN's LSTM
        
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 28, 28, dtype=torch.float32)

