import torch

data = torch.load("/home/drisspg/meta/scripts/sdpa/data/sdpa_nan_grad")
START = 0
BATCH_SIZE = -1
q = data["q"][START:BATCH_SIZE]
k = data["k"][START:BATCH_SIZE]
v = data["v"][START:BATCH_SIZE]
q.retain_grad()
k.retain_grad()
v.retain_grad()

print("Shapes: ", q.shape, k.shape, v.shape)
out_grad = data["out_grad"][START:BATCH_SIZE]
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
    attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out_tuple = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, attn_bias=None, compute_log_sumexp=True
    )
attended.backward(out_grad)
print("nan count Q:", q.grad.isnan().count_nonzero())
print("nan count K:", k.grad.isnan().count_nonzero())
print("nan count V: ", v.grad.isnan().count_nonzero())

q_index_min = tuple(map(lambda x: x.item(), torch.unravel_index(torch.argmin(q.grad), q.shape)))
k_index_min = tuple(map(lambda x: x.item(), torch.unravel_index(torch.argmin(k.grad), k.shape)))
v_index_min = tuple(map(lambda x: x.item(), torch.unravel_index(torch.argmin(v.grad), v.shape)))

print("q_index_min:", q_index_min)  # q_index_min: (2605, 2, 1, 0)
print("k_index_min:", k_index_min)  # k_index_min: (2605, 2, 0, 0)
print("v_index_min:", v_index_min)  # v_index_min: (2605, 2, 0, 0)

# Shape 2, 64
q_bad_head = q[2605, 2]
k_bad_head = k[2605, 2]
v_bad_head = v[2605, 2]

# This will contain subnormals
attn = q_bad_head @ k_bad_head.T
# regular softmax produces 1/num_sequence in this case 0.5
softmax = torch.nn.functional.softmax(attn, dim=-1)
output = softmax @ v

# Absmin values of the bad batch/head
a = torch.full((2, 64), 1.5414e-44, device="cuda", requires_grad=True).unsqueeze(1).unsqueeze(1)
b = torch.full((2, 64), 4.2039e-44, device="cuda", requires_grad=True).unsqueeze(1).unsqueeze(1)
c = torch.full((2, 64), 0.0018, device="cuda", requires_grad=True).unsqueeze(1).unsqueeze(1)
a.retain_grad()
b.retain_grad()
c.retain_grad()

out_tuple_again = torch.ops.aten._scaled_dot_product_efficient_attention(
    a, b, c, attn_bias=None, compute_log_sumexp=True
)

out_tuple_again[0].backward(out_tuple_again[0].clone())

assert q.grad.isnan().count_nonzero() == 0, "q.grad has NaNs!"


print("SUCCESS!✅")

tensor([0.6931,   -inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,
           inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,
           inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,
           inf,    inf,    inf,    inf,    inf], device='cuda:0')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.attention import SDPBackend, sdpa_kernel

embed_dim = 1024
batch_size = 32
seq_length = 50
num_iterations = 1000
learning_rate = 0.01
device = torch.device("cuda")
torch.autograd.set_detect_anomaly(True)

# Initialize model
model = nn.TransformerEncoderLayer( embed_dim, 16, dropout=0.01, batch_first=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
inputs = torch.full((seq_length, batch_size, embed_dim), 1000.0, device=device)

# Training loop
with sdpa_kernel(SDPBackend.MATH):
    for i in range(num_iterations):
        print(f'Iteration {i + 1}')
        optimizer.zero_grad()
        output = model(inputs)
        loss = output.mean()
        loss.backward()
        optimizer.step()