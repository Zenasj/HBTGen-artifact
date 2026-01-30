import torch
import torch.nn as nn

# Initialize embeddings
embedding = nn.Embedding(1000, 128)
anchor_ids = torch.randint(0, 1000, (1,), requires_grad=True)
positive_ids = torch.randint(0, 1000, (1,), requires_grad=True)
negative_ids = torch.randint(0, 1000, (1,), requires_grad=True)
anchor = embedding(anchor_ids)
positive = embedding(positive_ids)
negative = embedding(negative_ids)

# Built-in Distance Function
triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
output = triplet_loss(anchor, positive, negative)
output.backward()