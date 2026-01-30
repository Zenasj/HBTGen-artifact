for name, param in chain(model.named_parameters(), model.named_buffers()):
           print(name)

for name, _ in _iterate_valid_model_state(model):
      print(name)

import torch
import torch.nn as nn

class TiedEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TiedEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        self.decoder.weight = self.embedding.weight  # Tying weights

    def forward(self, input):
        embedded = self.embedding(input)
        output = self.decoder(embedded)
        return output

# Example usage
vocab_size = 10000
embedding_dim = 300
model = TiedEmbeddingModel(vocab_size, embedding_dim)

# Save model state_dict
torch.save(model.state_dict(), 'tied_embedding_model.pth')

# Load model state_dict
loaded_model = TiedEmbeddingModel(vocab_size, embedding_dim)
loaded_model.load_state_dict(torch.load('tied_embedding_model.pth'))