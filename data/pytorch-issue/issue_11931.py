import torch
weights = torch.randn(10000000, dtype=torch.float32).clamp(0.01, 1)

n_small = 100
n_big = 100000

# Works
sample_1 = torch.multinomial(weights, n_small, replacement=True)
print("Sampled 1")

# Works
sample_2 = torch.multinomial(weights, n_big, replacement=True)
print("Sampled 2")

# Works, but much slower
sample_3 = torch.multinomial(weights, n_small, replacement=False)
print("Sampled 3")

# I think it works, but it takes so long that it is unusable
sample_4 = torch.multinomial(weights, n_big, replacement=False)
print("Sampled 4")

p = torch.ones(1_000_000)
have = 0
want = 100_000
p_ = p.clone()
result = torch.empty(want, dtype=torch.long)
while have < want:
    a = torch.multinomial(p_, want-have, replacement=True)
    b = a.unique()
    result[have:have+b.size(-1)] = b
    p_[b] = 0
    have += b.size(-1)