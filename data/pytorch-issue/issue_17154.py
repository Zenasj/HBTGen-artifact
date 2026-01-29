# torch.rand(B, segments, C, dtype=torch.float32)
import torch
import torch.nn as nn

class SegmentConsensusModule(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, input_tensor):
        if self.consensus_type == 'avg':
            return input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            return input_tensor
        else:
            raise ValueError(f"Unsupported consensus_type: {self.consensus_type}")

class MyModel(nn.Module):
    def __init__(self, consensus_type='avg', dim=1):
        super().__init__()
        # Preserve original logic for 'rnn' -> 'identity' mapping
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        self.segment_consensus = SegmentConsensusModule(self.consensus_type, self.dim)

    def forward(self, x):
        return self.segment_consensus(x)

def my_model_function():
    # Default to 'avg' consensus_type as in original error context
    return MyModel(consensus_type='avg')

def GetInput():
    # Input shape (B, segments, C) matching SegmentConsensus's dim=1 expectation
    B, segments, C = 2, 5, 10
    return torch.rand(B, segments, C, dtype=torch.float32)

