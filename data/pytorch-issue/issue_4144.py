import torch
import torch.nn as nn

class RNN(nn.Module):
    """ RNN that handles padded sequences """

    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(RNN, self).__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, seq, seq_length):
        """ seq => [batch_size, seq_length, feats], seq_length => [batch_size]. Both are Variables """
        print(seq_length.size())
        # input()
        seq_length, idx_sort = torch.sort(seq_length, 0, descending=True)
        _, idx_unsort = idx_sort.sort(0)
        seq = seq[idx_sort]
        # print(seq_length)
        seq = pack_padded_sequence(seq, seq_length.data.cpu().numpy(), batch_first=True)
        o, s = self.gru(seq)
        s = s.squeeze()
        o, _ = pad_packed_sequence(o, batch_first=True)
        # print(o.squeeze().data.cpu().numpy())
        # print(s.squeeze().data.cpu().numpy())
        o = o[idx_unsort]
        s = s[idx_unsort]
        # print(o.squeeze().data.cpu().numpy())
        if self.bidirectional:
            s = torch.cat([s[0], s[1]], dim=-1)
        # print(o.size(), s.size())
        # input()
        return o, s