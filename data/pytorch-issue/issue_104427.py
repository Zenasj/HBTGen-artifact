# torch.rand(B, S, dtype=torch.long)  # Input shape: batch_size x sequence_length
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    class LoraAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.lora_A = nn.Linear(32, 8, bias=False)
            self.lora_B = nn.Linear(8, 32, bias=False)
            self.k_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)
            self.o_proj = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            q = self.q_proj(x)
            lora = self.lora_B(self.lora_A(q))
            combined = q + lora  # Simplified LoRA adaptation
            return combined  # Assume residual connection handled in decoder

    class LoraMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj1 = nn.Linear(32, 128, bias=False)
            self.proj2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            return self.proj2(F.gelu(self.proj1(x)))

    class LoraDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MyModel.LoraAttention()
            self.mlp = MyModel.LoraMLP()
            self.inp_layernorm = nn.LayerNorm(32)
            self.post_attn_layernorm = nn.LayerNorm(32)

        def forward(self, x):
            residual = x
            x = self.inp_layernorm(x)
            x = self.attn(x) + residual  # Residual connection
            x = self.post_attn_layernorm(x)
            mlp_in = x
            x = self.mlp(x) + mlp_in  # MLP residual
            return x

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)  # vocab_size=100, embedding_dim=32
        self.layers = nn.ModuleList([self.LoraDecoder() for _ in range(4)])
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, S = 2, 5  # Batch size and sequence length
    return torch.randint(0, 100, (B, S), dtype=torch.long)

