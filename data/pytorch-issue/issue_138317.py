import torch.nn as nn

import torch
from torch import Tensor, nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        attention_probs_dropout_prob: float = 0.1,
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

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )
        return attn_output


def test_attention():
    device = torch.device("cuda")
    num_attention_heads = 8
    hidden_size = 512
    attention_probs_dropout_prob = 0.0
    model = SelfAttention(
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
    ).to(device)

    model = torch.compile(model)
    batch_size = 8
    length = 1
    inputs_embeds = torch.randn(batch_size, length, hidden_size, device=device)
    attention_mask = torch.ones(batch_size, 1, length, length, device=device)
    attn_output = model(hidden_states=inputs_embeds, attention_mask=attention_mask)[0]
    loss = attn_output.mean()
    loss.backward()


if __name__ == "__main__":
    test_attention()