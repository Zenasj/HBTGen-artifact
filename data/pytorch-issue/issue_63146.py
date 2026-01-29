# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class Config:
    def __init__(self):
        self.hidden_size = 256  # Matches error context
        self.num_attention_heads = 8
        self.patch_size = 16
        self.num_hidden_layers = 12
        self.intermediate_size = 1024
        self.img_size = 224

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.head = nn.Linear(config.hidden_size, 10)  # num_classes=10 from issue context

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoded, _ = self.encoder(embedding_output)
        return self.head(encoded[:, 0])  # Use CLS token

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, config.hidden_size, 
                                         kernel_size=config.patch_size, 
                                         stride=config.patch_size)
        n_patches = (config.img_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embeddings
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attentions = []
        for layer in self.layer:
            hidden_states, attn = layer(hidden_states)
            attentions.append(attn)
        return self.norm(hidden_states), attentions

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, attn = self.attn(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual
        return x, attn

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim  # 256
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  # Fixed dimension
        
        self.out = nn.Linear(self.all_head_size, config.hidden_size)

    def forward(self, x):
        B = x.size(0)
        mixed_query = self.query(x)
        mixed_key = self.key(x)
        mixed_value = self.value(x)
        
        def reshape(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        query = reshape(mixed_query)
        key = reshape(mixed_key)
        value = reshape(mixed_value)
        
        scores = torch.matmul(query, key.transpose(-1,-2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1,2).contiguous().view(B, -1, self.all_head_size)
        return self.out(context), attn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def my_model_function():
    config = Config()
    return MyModel(config)

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

