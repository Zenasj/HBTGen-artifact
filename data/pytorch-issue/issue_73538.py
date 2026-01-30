import torch

# Creates some tensors in default dtype (here assumed to be float32)
print('Original dtypes')
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")
x_float16 = torch.rand((8, 8), device="cuda").half()

print(a_float32.dtype)
print(b_float32.dtype)
print(c_float32.dtype)
print(d_float32.dtype)

print('\nOutput dtypes within autocast enabled region')
with torch.autocast(device_type='cuda'):
    # torch.mm is on autocast's list of ops that should run in float16.
    # Inputs are float32, but the op runs in float16 and produces float16 output.
    # No manual casts are required.
    e_float16 = torch.mm(a_float32, b_float32)
    assert e_float16.dtype == x_float16.dtype
    # Also handles mixed input types
    f_float16 = torch.mm(d_float32, e_float16)
    assert e_float16.dtype == x_float16.dtype
    # torch.result_type is unaware of autocast's list and fail to compute the right dtype
    i_wrong_float = torch.result_type(a_float32, b_float32)
    assert i_wrong_float == x_float16.dtype
    j_wrong_float = torch.result_type(d_float32, e_float16)
    assert j_wrong_float == x_float16.dtype

print('\nOutput dtypes outside autocast region')
# After exiting autocast, calls f_float16.float() to use with d_float32
g_float32 = torch.mm(d_float32, f_float16.float())
assert g_float32.dtype == d_float32.dtype
# Outside autocast enabled region torch.result_type works as expected
h_float32 = torch.result_type(d_float32, f_float16.float())
assert h_float32 == d_float32.dtype