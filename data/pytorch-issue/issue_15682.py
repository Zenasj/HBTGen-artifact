import torch
d0 = torch.device('cuda:0')
d1 = torch.device('cuda:1')

with torch.cuda.device(d0):
   s0 = torch.cuda.current_stream()

with torch.cuda.device(d1):
    s1 = torch.cuda.current_stream()
    torch.cuda._sleep(1000000000)  # spin for about 1 sec on device1

print('everything finished (called on device 0)', s0.query(), s1.query())  # should be True, False
with torch.cuda.device(d1):
    print('everything finished (called on device 1)', s0.query(), s1.query())  # should be True, False