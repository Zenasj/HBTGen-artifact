import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=...) 
# Here, input shape is (65, 32, 128) as per the example in the issue
# dtype=torch.float32 is assumed based on the code's usage
# So: torch.rand(65, 32, 128, dtype=torch.float32)

class MyModel(nn.Module):
    def __init__(self, dim, heads, dim_feedforward, depth=8):
        super(MyModel, self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(TransformerModelLayer(dim, heads, dim_feedforward))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class KVTransform(nn.Module):
    def __init__(self, num_heads, head_dim):
        super(KVTransform, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x):
        n, b, _ = x.size()
        x = x.contiguous().view(n, b * self.num_heads, self.head_dim).transpose(0, 1)
        return x

class QTransform(nn.Module):
    def __init__(self, num_heads, head_dim, scaling):
        super(QTransform, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scaling

    def forward(self, x):
        n, b, _ = x.size()
        x = x * self.scaling
        x = x.contiguous().view(n, b * self.num_heads, self.head_dim).transpose(0, 1)
        return x

class AttnOutputTransform(nn.Module):
    def __init__(self, num_heads, dim):
        super(AttnOutputTransform, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

    def forward(self, x):
        n = x.size()[1]
        b = torch.div(x.size()[0], self.num_heads, rounding_mode="floor")
        x = x.transpose(0, 1).contiguous().view(n, b, self.dim)
        return x

class SelfAttentionFullModule(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, dropout_p=0.1):
        super(SelfAttentionFullModule, self).__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scaling = float(self.head_dim) ** 0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.to_out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(2)

        self.q_pre_transform = QTransform(self.num_heads, self.head_dim, self.scaling)
        self.kv_pre_transform = KVTransform(self.num_heads, self.head_dim)
        self.post_transform = AttnOutputTransform(self.num_heads, dim)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.q_pre_transform(q)
        k = self.kv_pre_transform(k)
        v = self.kv_pre_transform(v)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = self.post_transform(attn_output)
        attn_output = self.to_out(attn_output)
        return attn_output

class TransformerModelLayer(nn.Module):
    def __init__(self, dim, heads, dim_feedforward, dropout_p=0.1):
        super(TransformerModelLayer, self).__init__()
        self.self_attn = SelfAttentionFullModule(dim, heads=heads, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim_feedforward, dim),
        )

    def forward(self, x):
        x2 = self.self_attn(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.feedforward(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

def my_model_function():
    # Initialize MyModel with parameters from the issue example
    return MyModel(dim=128, heads=1, dim_feedforward=256, depth=2)

def GetInput():
    # Returns a random tensor matching the input shape (65, 32, 128)
    return torch.rand(65, 32, 128, dtype=torch.float32)

