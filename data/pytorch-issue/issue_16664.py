import torch
import torch.nn as nn

class Model(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.lstm = nn.LSTM(EMBEDDING_SIZE, LSTM_SIZE, num_layers=1, bidirectional=True)

    @jit.script_method
    def forward(self, tokens, seq_lengths):
        embedded = self.embedding(tokens)
        rnn_input = pack_padded_sequence(
            embedded,
            seq_lengths.int(), True, True,
        )
        rep, unused_state = self.lstm(rnn_input)
        unpacked, _ = pad_packed_sequence(
            rep,
            batch_first=True,
            padding_value=0.0,
            total_length=embedded.size(1),
        )
        return unpacked