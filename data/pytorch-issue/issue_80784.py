# torch.randint(0, 30000, (B, S), dtype=torch.long)  # B=batch_size, S=sequence_length
import torch
import torch.nn as nn

class FixedT5Attention(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance

    def _relative_position_bucket(self, relative_position, bidirectional=True):
        # Cast to int32 to avoid MPS int64 error
        relative_position = relative_position.to(torch.int)
        relative_buckets = 0
        if bidirectional:
            num_buckets = self.num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.int) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            zero = torch.zeros_like(relative_position)
            relative_position = -torch.min(relative_position, zero)
        # Bucketing logic (simplified for demonstration)
        # ... (original bucketing code with casts applied)
        return relative_buckets  # placeholder return for demonstration

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FixedT5Attention()

    def forward(self, input_ids):
        # Mimic T5's relative position computation with fix
        seq_length = input_ids.size(1)
        context_pos = torch.arange(seq_length, dtype=torch.int, device=input_ids.device)[:, None]
        mem_pos = torch.arange(seq_length, dtype=torch.int, device=input_ids.device)[None, :]
        rel_pos = mem_pos - context_pos
        # Apply fixed method
        _ = self.attention._relative_position_bucket(rel_pos, bidirectional=True)
        return input_ids  # Dummy output to maintain interface

def my_model_function():
    return MyModel()

def GetInput():
    # Random input_ids with token indices (int64 is unavoidable here but handled internally)
    return torch.randint(0, 30000, (1, 128), dtype=torch.long)

