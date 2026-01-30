import torch

torch.save(b'hello', '/tmp/dummy.pth')
torch.load('/tmp/dummy.pth', weights_only=False) # OK
torch.load('/tmp/dummy.pth', weights_only=True) # Error
# UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options 
# 	(1) Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
# 	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
# 	WeightsUnpickler error: Unsupported global: GLOBAL _codecs.encode was not an allowed global by default. Please use `torch.serialization.add_safe_globals([encode])` to allowlist this global if you trust this class/function.

# Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

torch.save(bytearray(b'hello'), '/tmp/dummy.pth')
torch.load('/tmp/dummy.pth', weights_only=False) # OK
torch.load('/tmp/dummy.pth', weights_only=True) # Error, same as above