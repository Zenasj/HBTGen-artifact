# Code to reproduce error
import torch
import torch.nn as nn
import random
from torch.nn import MSELoss


class RNNEncoder(nn.Module):
    def __init__(self, hid_dim=128, n_layers=2, dropout=0.5, device="cpu"):
        super().__init__()

        self.fc = nn.Linear(4, 128)
        self.dropout = nn.Dropout(dropout)

        self.fc_bn = nn.Linear(512, 128)

        self.rnn = nn.LSTM(input_size=128, hidden_size=hid_dim, batch_first=True, bidirectional=False,
                           num_layers=n_layers).to(device)

    def forward(self, x):
        h = self.fc(x)
        _, (hidden, cell) = self.rnn(h)
        return hidden, cell


class RNNDecoder(nn.Module):
    def __init__(self, input_dim=4, n_layers=2, hid_dim=128, dropout=0.5, device="cpu"):
        super().__init__()

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=n_layers)

        self.post_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )

    def forward(self, dec_input, hidden, cell):
        # Input shape: [batch_size, L, input_size]
        # Reshape to: [L, batch_size, input_size]
        dec_input = dec_input.reshape(dec_input.shape[1], -1, 4)

        rnn_out, (hidden, cell) = self.rnn(dec_input, (hidden, cell))

        # rnn_out = rnn_out.reshape(-1, 128, rnn_out.shape[0])
        rnn_out = rnn_out.reshape(-1, rnn_out.shape[0], 128)
        out = self.post_net(rnn_out)
        return out.flatten(1), (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device="cpu"):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def forward(self, bases, raw, teacher_forcing_ratio=0.5):

        batch_size = raw.shape[0]
        trg_len = raw.shape[1]

        outputs = torch.zeros(batch_size, trg_len).to(self.device)


        hidden, cell = self.encoder(bases)

        # Generate first input of zeros [batch_size, Length 1, input size]
        input_dec = torch.zeros((raw.shape[0], 1, 4)).to(self.device)

        for t in range(0, trg_len, 4):
            output_dec, (hidden, cell) = self.decoder(input_dec, hidden, cell)

            # Check if we are at the end of the target sequence and slice the required prediction length
            if t+4 > trg_len:
                pred_interval = trg_len - t
                outputs[:, t:t+pred_interval] = output_dec[:, :pred_interval]

            else:
                outputs[:, t:t+4] = output_dec

            teacher_force = random.random() < teacher_forcing_ratio

            input_dec = raw[:, t:t+4].unsqueeze(1) if teacher_force else output_dec.unsqueeze(1)

        return outputs


if __name__ == "__main__":
    device = "mps"
    encoder = RNNEncoder(device=device)
    decoder = RNNDecoder(device=device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device)

    loss = MSELoss()

    bases = torch.rand([32, 89, 4]).to(device)
    raw = torch.rand(32, 500).to(device)
    output = model(bases, raw)
    loss_fn = MSELoss()
    loss = loss(output, raw)
    loss.backward()

def forward(self, x):
        h = self.fc(x)
        _, (hidden, cell) = self.rnn(h)
        return hidden, cell

def forward(self, x):
        h = self.fc(x)
        output, (hidden, cell) = self.rnn(h)
        return hidden, cell + output.sum()