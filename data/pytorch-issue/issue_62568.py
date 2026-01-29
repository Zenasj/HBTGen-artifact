# torch.rand(5, 6, dtype=torch.float32)  # Inferred input shape based on the ONNX graph

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden2tag = nn.Linear(6, 3)
        self.lstm = nn.LSTM(6, 6, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Assuming x is of shape (seq_len, batch_size, input_size)
        h0 = torch.zeros(1, x.size(0), 6).to(x.device)
        c0 = torch.zeros(1, x.size(0), 6).to(x.device)
        
        # Reshape x to (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out.squeeze(1)
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space)
        
        return tag_scores

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(5, 6, dtype=torch.float32)

