import io
import torch

t = torch.rand(10)
b = io.BytesIO()

torch.save(t, b)
b.seek(0)
torch.load(b, map_location=torch.device("meta"))