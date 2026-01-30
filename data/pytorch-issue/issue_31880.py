import torch

t = torch.randn(3)
int8_t = t.char()
int8_t.requires_grad = True

int_tensor = torch.randint(0, 10, size=((4,4,3,3)), dtype=torch.uint8)
scale, zero_point = 1e-1, 0
q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)
q.requires_grad = True