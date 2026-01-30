import torch

# On Windows this writes crashes to C:\Users\<user>\AppData\pytorch_crashes
# On MacOS/Linux this writes crashes to /tmp/pytorch_crashes
torch.utils._crash_handler.enable_minidumps()

# Easy way to cause a segfault and trigger the handler
torch.bincount(input=torch.tensor([9223372036854775807]))