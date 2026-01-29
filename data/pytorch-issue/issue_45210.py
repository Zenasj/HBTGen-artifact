# Input is a tuple of three tensors each generated via torch.randint(0, 1000, (B,), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())

    def forward(self, inputs):
        anchor, positive, negative = inputs
        anchor_emb = self.embedding(anchor)
        positive_emb = self.embedding(positive)
        negative_emb = self.embedding(negative)
        return self.loss(anchor_emb, positive_emb, negative_emb)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate triplet input tensors without gradients as required
    return (
        torch.randint(0, 1000, (1,), dtype=torch.long),
        torch.randint(0, 1000, (1,), dtype=torch.long),
        torch.randint(0, 1000, (1,), dtype=torch.long)
    )

