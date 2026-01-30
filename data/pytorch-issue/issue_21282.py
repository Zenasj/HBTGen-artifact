import torch.nn as nn

from torch import nn
import torch.jit

class MyLSTM(torch.jit.ScriptModule):
    def __init__(self):

        super(MyLSTM, self).__init__()
        
        self.rnn = nn.LSTM(input_size=300,
                           hidden_size=32,
                           batch_first=True,
                           num_layers=1,
                           bidirectional=False)
    
    @torch.jit.script_method
    def forward(self, x_in, x_lengths):
        max_length = x_lengths[0]
        x_in_shortened = x_in[:,:max_length]
        
        x_packed = nn.utils.rnn.pack_padded_sequence(x_in_shortened, 
                                                     x_lengths,
                                                     True, # batch_first
                                                     True # enforce_sorted
                                                    )
        
        lstm_outputs = self.rnn(x_packed)
        sequence_outputs = nn.utils.rnn.pad_packed_sequence(lstm_outputs[0], 
                                                            batch_first=True, 
                                                            padding_value=0.0, 
                                                            total_length=max_length)
        
        lstm_out = sequence_outputs[0]
        out_seq_length = sequence_outputs[1]

        return lstm_out

# Instantiation:
my_lstm = MyLSTM()

x_packed = nn.utils.rnn.pack_padded_sequence(x_in_shortened, 
                                             x_lengths,
                                             torch.ones(1).byte().squeeze(), # batch_first
                                             torch.ones(1).byte().squeeze() # enforce_sorted
                                            )