import torch

x = torch.ones(3, device="cpu")
# At this point, `x` is filled with 1
y = torch.rand(3, device="cpu")
# At this point, `y` contains 3 randomly generated values
z = x + y
# At this point, `z` contains result of element-wise sum of `x` and `y`
print(z) # At this just prints the memory contents

x = torch.ones(3, device="mps")
# At this point, 12 bytes of GPU memory are allocated, `x.data_ptr()` points to it and command to fill this memory with ones is submitted to command queue (that might or might not execute in the background)
y = torch.rand(3, device="mps")
# Same as before, memory allocated, `y.data_ptr()` references it and command to fill it with random values has been queued and guaranteed to be executed after the first command
z = x + y
# At this point, `z` points to GPU memory, command to perform element wise sum has been issued
print(z)
# To print values, they must be copied to CPU, which triggers sync (i.e. commit and wait)
# That waits for all the previous operations to complete