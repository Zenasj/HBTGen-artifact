# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, time_steps)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers):
        super(MyModel, self).__init__()
        self.lstm_layer = nn.GRU(vocab_size, num_hiddens, num_layers, batch_first=True)
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm_layer.num_layers, x.size(0), self.lstm_layer.hidden_size).to(x.device)
        out, _ = self.lstm_layer(x, h0)
        out = self.fc(out)
        return out

def my_model_function():
    vocab_size = 1000  # Example vocab size, adjust as needed
    num_hiddens = 256
    num_layers = 1
    return MyModel(vocab_size, num_hiddens, num_layers)

def GetInput():
    batch_size = 256
    time_steps = 35
    vocab_size = 1000  # Example vocab size, adjust as needed
    input_tensor = torch.randint(0, vocab_size, (batch_size, time_steps))
    input_tensor = torch.nn.functional.one_hot(input_tensor, num_classes=vocab_size).float()
    return input_tensor

