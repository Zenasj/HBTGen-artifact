import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, features, num_hiddens, num_layers):
        super(LSTMNet, self).__init__()
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.encoder = nn.LSTM(input_size=features, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                batch_first=True)
                                # bidirectional=True)
        self.decoder = nn.Linear(num_hiddens, 1) # 4 * as default, try 1*
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        inputs = inputs.view(len(inputs), 1, -1)
        outputs, _ = self.encoder(inputs) # (seq_len, batch_size, 2*h)
        encoding = outputs.view(-1, self.num_hiddens)
        outs = self.decoder(encoding) # (batch_size, 2)
        return outs

class FCNet(nn.Module): 
    def __init__(self, features):
        super(FCNet, self).__init__()        
        self.linear_relu1 = nn.Linear(features, 64)
        self.linear_relu2 = nn.Linear(64, 128)
        self.linear_relu3 = nn.Linear(128, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 1)        
    def forward(self, x):        
        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)
        y_pred = self.linear5(y_pred)
        return y_pred