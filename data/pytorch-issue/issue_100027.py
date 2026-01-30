import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self, num_heads, hidden_dim):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query = torch.nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.scale = torch.nn.Parameter(torch.FloatTensor([hidden_dim // num_heads]).sqrt())

    def forward(self, x):
        (b, s, h) = x.size()
        query = self.query(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        key = self.key(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        value = self.value(x).view(b, s, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) / self.scale
        attn_probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, value)
        return attn_output

num_heads = 1
hidden_dim = 512
sequence_length = 16
batch_size = 4

x = torch.randn(batch_size, sequence_length, hidden_dim)
func = Model(num_heads, hidden_dim).to('cpu')

res1 = func(x)
print(res1)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # RuntimeError: aten::scaled_dot_product_attention() Expected a value of type 'Optional[float]' for argument 'scale' but instead found type 'FakeTensor'.
    # Position: 6
    # Value: FakeTensor(..., size=(1,))
    # Declaration: aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> Tensor
    # Cast error details: Unable to cast Python instance to C++ type (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)

output, _ = MHA(..., need_weights=True)