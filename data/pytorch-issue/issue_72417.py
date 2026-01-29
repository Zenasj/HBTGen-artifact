# torch.randint(0, 128, (B, S), dtype=torch.long)  # B=batch, S=sequence length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=100, hidden_dim=50, vocab_size=128):
        super(MyModel, self).__init__()
        self.initrange = 0.1
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.num_labels = num_classes  # Assuming num_classes is a list like [38]

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.uniform_(-self.initrange, self.initrange)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Linear(2 * hidden_dim, self.num_labels[0])  # 2x hidden for bidirectional
        self.init_weights(self.classifier)  # Initialize classifier weights

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.data.uniform_(-self.initrange, self.initrange)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Embedding):
            layer.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, (h_n, _) = self.lstm(embeds)
        
        # Extract final hidden states
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)  # 2 directions
        final_layer_hn = h_n[-1]  # Last layer's outputs (shape: 2, batch, hidden_dim)
        
        # Concatenate forward and backward hidden states
        forward_h = final_layer_hn[0]  # (batch, hidden_dim)
        backward_h = final_layer_hn[1]  # (batch, hidden_dim)
        final_hidden = torch.cat((forward_h, backward_h), dim=1)  # (batch, 2*hidden_dim)
        
        # Apply classifier
        logits = self.classifier(final_hidden)
        return logits

def my_model_function():
    # Returns a model instance with default parameters from original code
    return MyModel(num_classes=[38], embedding_dim=100, hidden_dim=50, vocab_size=128)

def GetInput():
    # Generate random input tensor with shape (batch_size, seq_length)
    batch_size = 10
    seq_length = 10
    vocab_size = 128
    return torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)

