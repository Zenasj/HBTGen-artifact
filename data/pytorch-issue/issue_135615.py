import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttentionModel(nn.Module):
    def __init__(self, d_model, scale):
        super(ScaledDotProductAttentionModel, self).__init__()
        self.scale = scale  # scaling factor for attention scores
        self.d_model = d_model  # dimensionality of input embeddings
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, query_states, key_states, value_states):
        # Project the input states
        query = self.query_linear(query_states)
        key = self.key_linear(key_states)
        value = self.value_linear(value_states)

        # Perform scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value, scale=self.scale
        )
        return attn_output

d_model = 64
scale = 1.0 / (d_model ** 0.5)
model = ScaledDotProductAttentionModel(d_model, scale)

batch_size = 2
seq_length_q = 10  # length of query
seq_length_kv = 15  # length of key and value
embedding_dim = d_model

query_states = torch.randn(batch_size, seq_length_q, embedding_dim)
key_states = torch.randn(batch_size, seq_length_kv, embedding_dim)
value_states = torch.randn(batch_size, seq_length_kv, embedding_dim)

output = model(query_states, key_states, value_states)
print(output.shape)

onnx_file_path = "scaled_dot_product_attention.onnx"
torch.onnx.export(
    model,  # model being exported
    (query_states, key_states, value_states),  # example input (tuple)
    onnx_file_path,  # where to save the ONNX model
    input_names=["query_states", "key_states", "value_states"],  # input names
    output_names=["attn_output"],  # output names
)

print(f"Model exported to {onnx_file_path}")

attn_output = F.scaled_dot_product_attention(
            query, key, value, scale=self.scale
        )

attn_output = F.scaled_dot_product_attention(
            query, key, value
        )