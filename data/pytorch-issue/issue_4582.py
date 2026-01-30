import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, embedding_dim, dropout_prob=0, max_length=30):
        super(Model, self).__init__()
        
        self.hidden_dim = embedding_dim
        self.max_length  = max_length
        
        self.num_layers = 3
        self.rnn = nn.GRU(input_size=in_features, hidden_size=embedding_dim,
                          num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)

    def forward(self, x, seq_lengths):
        # Clamp everything to minimum length of 1, but keep the original variable to mask the output later
        seq_lengths_clamped = seq_lengths.clamp(min=1, max=self.max_length)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths_clamped, 
                                                           enforce_sorted=False, batch_first=True)  
        out_packed, hidden = self.rnn(x_packed)
        # hidden is of the shape num_layers * num_directions, batch, hidden_size
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)        

        # SOME TRANSFORMATIONS HERE
        # get the last element i.e from the last layer, rearrange shape to be batch, num_directions, hidden_size
        hidden = hidden[-1].permute(1, 0, 2)
        # since it's bidirectional, we combine both outputs from last layer
        hidden = hidden.reshape(-1, 2 * self.hidden_dim)

       # MASKING HERE
        # mask everything that had seq_length as 0 in input as 0
        hidden.masked_fill_((seq_lengths == 0).view(-1, 1), 0)
        return hidden