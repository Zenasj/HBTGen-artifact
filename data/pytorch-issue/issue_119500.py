import torch

# in this repro, "grad_out" and "value" are transposed tensors,
# but "key" and "value" are contiguous
a = torch.randn(2, 513, 16, 64, dtype=torch.float16, device='cuda').transpose(1, 2)
b = torch.randn(2, 16, 513, 64, dtype=torch.float16, device='cuda')
c = torch.randn(2, 16, 513, 64, dtype=torch.float16, device='cuda')
d = torch.randn(2, 513, 16, 64, dtype=torch.float16, device='cuda').transpose(1, 2)
e = torch.randn(2, 16, 513, 64, dtype=torch.float16, device='cuda')
f = torch.randn(2, 16, 513, device='cuda')
g = None
h = None
i = 513
j = 513
k = 0.0
l = False
m = torch.tensor(1, dtype=torch.int64)
n = torch.tensor(1, dtype=torch.int64)

out1_ref, out2_ref, out3_ref = torch.ops.aten._scaled_dot_product_flash_attention_backward(a, b, c, d, e, f, g, h, i, j, k, l, m, n, scale=0.125)

from torch._meta_registrations import meta__scaled_dot_product_flash_backward
out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(a, b, c, d, e, f, g, h, i, j, k, l, m, n, scale=0.125)

# prints True True
print(out1_ref.is_contiguous())
print(out1_test.is_contiguous())

# prints True True
print(out2_ref.is_contiguous())
print(out2_test.is_contiguous())

# prints True False
print(out3_ref.is_contiguous())
print(out3_test.is_contiguous())