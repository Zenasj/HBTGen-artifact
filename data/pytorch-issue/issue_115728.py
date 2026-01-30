import torch
import torch.nn as nn

class GPT2Block(nn.Module):

    def __init__(self, config, window_size):
        super().__init__()
        self.mp_size = iint(os.getenv("WORLD_SIZE", "1"))
        self.hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.attn = GPT2Attention(config, window_size)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states, attention_mask, past_kv, kv_len, wpe=None):
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(hidden_states, attention_mask, past_kv, kv_len, wpe)
        mlp_output = self.mlp(hidden_states)

        layer_out = attn_output + mlp_output

        if self.mp_size > 1:
            torch.distributed.all_reduce(layer_out)

        layer_out = layer_out + residual
        return layer_out

torch.compile(self.transformer, dynamic=True, fullgraph=True)