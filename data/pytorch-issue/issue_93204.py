# torch.rand(B, S, dtype=torch.long)  # B=batch_size, S=sequence_length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32000, 512)  # T5-small vocab and hidden size
        self.attention_proj = nn.Linear(512, 512)
        self.bidirectional = True  # Parameter causing Dynamo tracing issues
        
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # Simplified version of T5's _relative_position_bucket function
        # Dynamo struggles with multiple parameter values for 'bidirectional'
        ret = torch.zeros_like(relative_position)
        if bidirectional:
            num_buckets //= 2
            ret[relative_position > 0] = num_buckets
            relative_position = torch.abs(relative_position)
        # Dummy implementation to trigger Dynamo tracing issues
        return ret % num_buckets
    
    def forward(self, x):
        # Simulate T5 attention mechanism with problematic bucketing function
        x = self.embedding(x)
        positions = torch.arange(x.size(1), device=x.device)
        relative_pos = positions[None, :] - positions[:, None]
        buckets = self._relative_position_bucket(
            relative_pos,
            bidirectional=self.bidirectional,
            num_buckets=32,
            max_distance=128
        )
        # Dummy attention computation using buckets
        attn_output = self.attention_proj(x)
        return attn_output

def my_model_function():
    # Returns model instance with parameters initialized
    return MyModel()

def GetInput():
    # Returns random input matching T5's expected input (input_ids)
    return torch.randint(0, 32000, (8, 128), dtype=torch.long)

