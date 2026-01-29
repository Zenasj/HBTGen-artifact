# torch.rand(B, 1, 32, 320, dtype=torch.float32)  # Inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=1000, nh=256, n_rnn=2):
        super(MyModel, self).__init__()
        assert imgH % 16 == 0, "imgH must be a multiple of 16"
        
        # CNN configuration
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        
        self.cnn = nn.Sequential()
        def convRelu(i, batch_norm=False):
            nIn = nc if i == 0 else nm[i-1]
            nOut = nm[i]
            self.cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batch_norm:
                self.cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            self.cnn.add_module(f'relu{i}', nn.ReLU(True))
        
        # Build CNN layers
        convRelu(0)
        self.cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)
        self.cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2)
        convRelu(3)
        self.cnn.add_module('pooling2', nn.MaxPool2d((2,2), (2,1), (0,1)))
        convRelu(4, batch_norm=True)
        convRelu(5, batch_norm=True)
        self.cnn.add_module('pooling5', nn.MaxPool2d((2,2), (2,1), (0,1)))
        convRelu(6, batch_norm=True)
        
        # RNN: LSTM with bidirectional layers
        self.rnn = nn.LSTM(
            input_size=nm[-1],  # 512 channels from CNN
            hidden_size=nh,
            num_layers=n_rnn,
            bidirectional=True,
            batch_first=False
        )
        
        # Final classification layer
        self.linear = nn.Linear(nh * 2, nclass)  # 2*nh due to bidirectional

    def forward(self, x):
        # CNN processing
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1, "CNN output height must be 1"
        
        # Reshape for RNN (seq_len, batch, channels)
        x = x.squeeze(2).permute(2, 0, 1)  # (width, batch, channels)
        
        # RNN processing
        x, _ = self.rnn(x)
        
        # Linear layer
        x = self.linear(x)
        return x

def my_model_function():
    return MyModel(nh=256)  # Use nh=256 as specified in user's parameters

def GetInput():
    # Generate random input tensor matching model's input requirements
    return torch.rand(128, 1, 32, 320, dtype=torch.float32)

