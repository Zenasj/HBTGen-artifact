import torch

torch.save(b'hello', '/tmp/dummy.pth')

torch.serialization.add_safe_globals([_codecs.encode])
torch.load('/tmp/dummy.pth', weights_only=True)