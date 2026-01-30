import torch

# bench.py
x = torch.zeros(1, device='cpu', requires_grad=True)

start_fw = time.time()
for _ in range(100000):
    x = x * x
end_fw = time.time()

start_bw = time.time()
x.backward()
end_bw = time.time()
print("Forward\t", end_fw-start_fw, end="\t")
print("Backward\t", end_bw-start_bw)