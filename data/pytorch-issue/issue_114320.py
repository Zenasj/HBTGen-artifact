import torch
print(torch.__version__)

torch.cuda.empty_cache()
x=torch.randn(10240000, device="cuda")
y=torch.rand_like(x)
g=torch.cuda.CUDAGraph()
s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
s0.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s0):
  g.capture_begin()
  z = x+y
  with torch.cuda.stream(s1):
    s1.wait_stream(s0)
    w = z+y
  s0.wait_stream(s1)
  g.capture_end()
segments = torch.cuda.memory_snapshot()
for s in segments:
  print("POOL ID", s["segment_pool_id"], s["active_size"], s["total_size"], s["stream"])