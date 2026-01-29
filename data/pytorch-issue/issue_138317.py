# torch.rand(B, L, H, dtype=torch.float32) and attention_mask of shape (B, 1, L, L)
import torch
from torch import nn, Tensor

class MyModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        hidden_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout_prob = attention_probs_dropout_prob

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(new_x_shape).permute(0, 2, 1, 3)

    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        return torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 8
    length = 1
    hidden_size = 512
    hidden_states = torch.randn(batch_size, length, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, 1, length, length, dtype=torch.float32)
    return (hidden_states, attention_mask)

