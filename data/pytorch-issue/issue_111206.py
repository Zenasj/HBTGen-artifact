# torch.randint(0, 100, (B, S), dtype=torch.long)
import torch
from torch import nn
import torch.utils.checkpoint

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([GPTBigCodeAttention(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None):
        # Create attention_mask with custom attributes
        attention_mask = torch.Tensor([4, 5, 6])  # Fixed tensor for reproducibility
        attention_mask.huhu = "is this working?"
        attention_mask._padding_mask = None

        hidden_states = self.wte(input_ids)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # Only checkpoint the first attention layer for simplicity
        outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.h[0]),
            hidden_states,
            attention_mask,
        )
        return outputs

class GPTBigCodeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.hidden_size, config.hidden_size + 2 * 15)

    def forward(self, hidden_states, attention_mask):
        # Directly access attributes to trigger the error during recompute
        print("attention_mask.huhu", attention_mask.huhu)
        print("attention_mask._padding_mask", attention_mask._padding_mask)
        query = self.c_attn(hidden_states)
        return query

def my_model_function():
    class DummyConfig:
        hidden_size = 32
        vocab_size = 100
        num_hidden_layers = 2
    return MyModel(DummyConfig())

def GetInput():
    # Returns input_ids tensor of shape (batch=1, sequence_length=4)
    return torch.randint(0, 100, (1, 4), dtype=torch.long)

