import torch

file = open('/tmp/big.bin', 'rb')
file_bytes = file.read()
mytensor = torch.frombuffer(file_bytes, dtype=torch.float32).reshape(a, b)