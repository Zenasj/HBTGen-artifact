import torch
import torch.nn as nn

class RnnTest(nn.Module):
    def __init__(self):
        super(RnnTest, self).__init__()
        self.rnn = nn.LSTM(3, 4, batch_first=True, bias=True)
        self.fc = nn.Linear(4, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
rnn = RnnTest()
rx = torch.rand(1, 10, 3)
rnn(rx)
torch.onnx.export(rnn, rx, 'rnn.onnx', verbose=True, output_names=['output'])

class RnnTest(nn.Module):
    def __init__(self):
        super(RnnTest, self).__init__()
        self.rnn = nn.LSTM(3, 4, batch_first=True, bias=False)
        self.fc = nn.Linear(4, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
rnn = RnnTest()
rx = torch.rand(1, 10, 3)
rnn(rx)
torch.onnx.export(rnn, rx, 'rnn.onnx', verbose=True, output_names=['output'])