import torch

# 1. Make a base tensor and save the data address.
base = torch.cuda.FloatTensor([10, 10])
data_ptr = base.data_ptr()

# 2. Make a view tensor with storage_offset() > 0.
view = base[5:]

# 3. Compute something on the view tensor in a custom stream.
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    torch.cuda._sleep(50000000)

# 4. Record the stream on the view tensor.
view.record_stream(stream)

# 5. Delete relevant tensors.
del base, view
torch.cuda.current_stream().synchronize()

# 6. Make a new tensor when the computations in the stream are not finished yet.
try_realloc = torch.cuda.FloatTensor([10, 10])

# 7. The base storage may be reallocated to the new tensor.
assert try_realloc.data_ptr() != data_ptr  # It fails!