# torch.rand(B, S, dtype=torch.long)  # B=batch, S=sequence length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, batch_size=10, embedding_dim=100, hidden_dim=50, vocab_size=128):
        super(MyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)  # 2x for bidirectional
        
        # Initialize embeddings uniformly
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, sentence, labels=None):
        embeds = self.word_embeddings(sentence)
        lstm_out, (hidden, _) = self.lstm(embeds)
        
        # Extract final hidden states from both directions of last layer
        num_directions = 2 if self.lstm.bidirectional else 1
        final_hidden = torch.cat(
            [hidden[-num_directions + i] for i in range(num_directions)], 
            dim=1
        )
        
        logits = self.classifier(final_hidden)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.view(-1))
            return loss, logits
        return logits

def my_model_function():
    return MyModel(num_classes=38)  # Matches reported self.num_labels[0] = 38 in error logs

def GetInput():
    # Generate random input with batch=10 (default) and sequence length 20
    return torch.randint(0, 128, (10, 20), dtype=torch.long)

