import torch

t = torch.tensor([complex(0, float('inf'))] * 4)
out_non_vec = torch.sgn(t)
ref = torch.tensor([complex(float('nan'), float('nan'))] * 4)

print(out_non_vec)
are_equal, msg = torch.testing._compare_tensors_internal(out_non_vec, ref, equal_nan=True, rtol=0, atol=0)
print("EQUAL:", are_equal)

t = torch.tensor([complex(0, float('inf'))] * 9)
out_vec = torch.sgn(t)
ref = torch.tensor([complex(float('nan'), float('nan'))] * 9)

print(out_vec)
are_equal, msg = torch.testing._compare_tensors_internal(out_vec, ref, equal_nan=True, rtol=0, atol=0)
print("EQUAL:", are_equal)