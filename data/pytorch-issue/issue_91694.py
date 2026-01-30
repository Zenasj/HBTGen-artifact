import torch
import torch.nn as nn

class BasciRNN(nn.Module):

    def __init__(self, d_input, n_hidden,
                 n_layers, n_output,
                 dropout):
        super(BasciRNN, self).__init__()
        self.d_input = d_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_output = n_output
        self.drop = dropout

        self.rnn = nn.LSTM(input_size=self.d_input,
                           hidden_size=self.n_hidden,
                           num_layers=self.n_layers,
                           batch_first=True,
                           dropout = self.drop,
                           )

        self.fc = nn.Linear(self.n_hidden, self.n_output)


    def forward(self, x):

        h0 = torch.zeros(self.n_layers,
                         x.size(0), self.n_hidden).to(device)
        c0 = torch.zeros(self.n_layers,
                         x.size(0), self.n_hidden).to(device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:,-1,:])

        return out

emb_len=16
model = BasciRNN(d_input = emb_len, n_hidden = 12,
                 n_layers = 8, n_output = 1,
                 dropout = 0.2)

print("Using", device)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10
batch=64
seq_len=32
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    X_batch = torch.randn((batch, seq_len, emb_len)).to(device)
    y_batch = torch.randn((batch)).clamp(0, 1).to(device)

    model_outputs = model(X_batch).squeeze()
    loss = criterion(model_outputs, y_batch)
    print(loss.item())

    loss.backward()
    optimizer.step()