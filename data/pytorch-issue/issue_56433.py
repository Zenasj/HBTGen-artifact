# torch.rand(seq_len, batch_size, 512, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(512, 512, num_layers=2, dropout=0.5, batch_first=False)

    def forward(self, x):
        return self.lstm(x)[0]  # Return outputs only for simplicity

def my_model_function():
    model = MyModel()
    return model.cuda()  # Explicitly move to CUDA as in the issue's example

def GetInput():
    # Matches LSTM input requirements (seq_len, batch, input_size)
    return torch.rand(10, 1, 512, dtype=torch.float32).cuda()

