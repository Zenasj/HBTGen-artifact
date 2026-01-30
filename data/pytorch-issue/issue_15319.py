import torch
import torch.nn as nn

class FeedRNN(torch.jit.ScriptModule):
    def __init__(self):
        super(FeedRNN, self).__init__()
        self.lstm = nn.LSTM(30, 25, batch_first=True)

    @torch.jit.script_method
    def forward(self, X):
        # type: (Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]])
        return self.lstm(X)


feed_rnn = FeedRNN()

X = nn.utils.rnn.pack_padded_sequence(b, b_lengths, batch_first=True)
X, hidden_states = feed_rnn(X)
print(X, hidden_states)