import torch

s0 = torch.cuda.current_stream()
s1 = torch.cuda.Stream(device=1)

with torch.cuda.stream(s1):
    torch.cuda._sleep(500000000)

print(s0.query())  # prints False
print(s1.query())  # prints True