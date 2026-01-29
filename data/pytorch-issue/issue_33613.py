# torch.randint(5000, (B, 50), dtype=torch.long)
import torch
import torch.nn as nn

class State:
    def __init__(self):
        self.ntoken = 5000  # Vocabulary size
        self.ninp = 50      # Embedding dimension
        self.num_classes = 2  # IMDB has 2 classes
        self.pretrained_vec = torch.rand(self.ntoken, self.ninp)  # Random pretrained embeddings
        self.learnable_embedding = True  # Whether embeddings are trainable
        self.textdata = True  # Input is text data

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.state = State()  # Internal state parameters
        
        self.output_dim = 1 if self.state.num_classes == 2 else self.state.num_classes
        embedding_dim = self.state.ninp
        ntoken = self.state.ntoken
        nhead = 1
        hidden_dim = embedding_dim
        n_layers = 1
        dropout = 0.1
        
        # Embedding layer
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(self.state.pretrained_vec)
        self.embed.weight.requires_grad = self.state.learnable_embedding
        
        # Transformer decoder components
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Matches input tensor dimensions
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        
        # Output layer
        self.classifier_head = nn.Linear(hidden_dim, self.output_dim)
        self.distilling_flag = False  # Disable distilling mode by default

    def forward(self, x):
        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x)  # Embed input tokens
            else:
                out = torch.squeeze(x)
        else:
            out = x  # Direct input if not text
            
        # Create target sequence (required for TransformerDecoder)
        tgt_size = list(out.size())
        tgt_size[-2] = 1  # Set target sequence length to 1
        tgt = torch.rand(tgt_size, dtype=out.dtype, device=out.device)
        
        # Decoder expects (batch, sequence, features) with batch_first=True
        hidden = self.decoder(tgt, out).squeeze(1)
        return self.classifier_head(hidden)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    seq_len = 50  # Max sequence length from --maxlen 50
    return torch.randint(5000, (B, seq_len), dtype=torch.long)

