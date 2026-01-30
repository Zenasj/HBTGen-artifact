import torch
print(torch.__version__)

# Normal behavior
a = torch.tensor([[5000]]).float()
b = torch.tensor([[6500]]).float()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print(a)
print(b)
print(d1)
print(d2)

print()

# Buggy behavior
a = torch.tensor([[512695]]).float()
b = torch.tensor([[512804]]).float()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print(a)
print(b)
print(d1)
print(d2)

a = torch.tensor([[512695]]).double()
b = torch.tensor([[512804]]).double()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

a = torch.tensor([[386784556587]]).double()
b = torch.tensor([[386783820152]]).double()
d1 = torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
d2 = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")

print(a)
print(b)
print(a - b)
print(d1)
print(d2)

tensor([[3.8678e+11]], dtype=torch.float64)
tensor([[3.8678e+11]], dtype=torch.float64)
tensor([[736435.]], dtype=torch.float64)
tensor([[736448.9539]], dtype=torch.float64)
tensor([[736435.]], dtype=torch.float64)