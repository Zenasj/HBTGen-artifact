import torch

vocab_size = BIGNUM # for example, 1000000
active_size = SMALLNUM # for example, 3000
active_vocab = torch.randperm(vocab_size)[:SMALLNUM].long()
index = active_vocab.argsort()
lookup = torch.zeros(BIGNUM).fill_(SMALLNUM+1).long()
lookup[active_vocab] = index

# A tensor including only indices from the "active vocab"
some_tensor = active_vocab[torch.LongTensor(1000).random_(0, active_size)]

# Your dense embedding matrix.
embedding = torch.FloatTensor(active_size, 400)

# Map your tensor's indices to those corresponding with the dense embeddings.
embedding_lookup = lookup[some_tensor]

# Sparse Embedding Lookup with Dense Embedding
output = embedding[embedding_lookup]