# torch.rand(128, 64, 768, dtype=torch.float32, device='cuda:0')  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_per_head=64, base=10000, max_seq_length=128, seq_dim: int = 0):
        super().__init__()
        self.hidden_size = 768
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.seq_dim: int = seq_dim
        freqs_cis = self.precompute_freqs_cis(dim=hidden_per_head, end=max_seq_length * 2, theta=base)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, hidden_states):
        mixed_x_layer = self.query_key_value(hidden_states)
        mixed_x_layer = mixed_x_layer.view(hidden_states.shape[0], hidden_states.shape[1], 12, 3 * 64)

        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [64] * 3, dim=3)

        query_layer, key_layer = self.apply_rotary_emb(query_layer, key_layer)

        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2))

        return matmul_result

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(self.freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        freqs_cis = freqs_cis[: x.shape[self.seq_dim]]
        shape = [s if i == self.seq_dim or i == x.ndim - 1 else 1 for i, s in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(128, 64, 768, dtype=torch.float32, device='cuda:0', requires_grad=True)

