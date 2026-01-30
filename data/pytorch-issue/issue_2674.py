import torch
import torch.nn as nn

class SymbolModule(nn.Module):
    # MINIBATCH in: [64, 150]
    # LABEL: [64]
    def __init__(self):
        super(SymbolModule, self).__init__()
        # Output: (mini-batch, W, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=MAXFEATURES,
                                      embedding_dim=EMBEDSIZE)
        self.lstm = nn.LSTM(input_size=EMBEDSIZE, 
                            hidden_size=NUMHIDDEN)
        self.l_out = nn.Linear(NUMHIDDEN, 2)

    def forward(self, x):
        embeds = self.embedding(x)
        #print(embeds.size(0), embeds.size(1), embeds.size(2))
        # (64, 150, 125)
        _batch_size = embeds.size(1)  # 150
        # init hidden-state for each element in batch
        h0 = Variable(torch.zeros(1, _batch_size, NUMHIDDEN).cuda())
        # init cell-state for each element in batch
        c0 = Variable(torch.zeros(1, _batch_size, NUMHIDDEN).cuda())
        # output for each t, (last_hidden-state, last_cell-state)
        lstm_out, (hn, cn) = self.lstm(embeds, (h0, c0))
        #print(hn.size(0), hn.size(1))
        # (1, 150)
        hn1 = hn.transpose(0, 1).contiguous().view(_batch_size, -1)
        #print(hn1.size(0))
        # 150
        out = self.l_out(hn1)
        #print(out.size(0), out.size(1))
        # (150, 2)
        return out

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
(25000, 150) (25000, 150) (25000,) (25000,)

for j in range(EPOCHS):
    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):
        # Get samples
        data = Variable(torch.LongTensor(data).cuda())
        target = Variable(torch.LongTensor(target).cuda())
        # Init
        optimizer.zero_grad()
        # Clear out hidden state of lstm
        sym.hidden = sym.init_hidden()
        # Forwards
        output = sym(data)
        # Loss
        loss = criterion(output, target)
        # Back-prop
        loss.backward()
        optimizer.step()