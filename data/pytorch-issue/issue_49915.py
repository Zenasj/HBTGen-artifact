3
import torch

# We will use GPU-0 and GPU-1.
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)

to_copy = torch.ones(100, device=0)

# Introduce a separate stream for copy and synchronize to the default stream.
# The copy will be started when "to_copy" is ready.
copy_stream = torch.cuda.Stream(0)
copy_stream.wait_stream(torch.cuda.default_stream(0))

with torch.cuda.stream(copy_stream):
    torch.cuda._sleep(100000000)
    copied = to_copy.to(1)

    # Try to uncomment any one of following two lines and rerun.
    #print(copied[0])
    #copied.sum().item()

    to_copy = None

# Here's any computation which allocates some new tensors on the default stream.
other = torch.rand(100, device=0)

# It will be 100 or about 50 depending on whether you comment print(copied[0]) or copied.sum().item().
print(copied.sum().item())