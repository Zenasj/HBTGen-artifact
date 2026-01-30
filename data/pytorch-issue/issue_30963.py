import torch.nn as nn

net=nn.Embedding(200,300)

for n,p in net.named_parameters():
    print(n,p)#   get result
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0.):
        super(Encoder, self).__init__()
        self.embedd = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        embedding = self.embedd(inputs.long()).permute(1, 0, 2)
        return self.rnn(embedding)
encoder = Encoder(20, 64, 64, 2,0.5)
print(encoder)
def init_weights(m):
    print(m)
    for name,param in m.named_paramters():
        init.uniform_(param.data,-0.08,0.08)

encoder.apply(init_weights)

net=nn.Embedding(200,300)

for n,p in net.named_parameters():
    print(n,p)#   get result