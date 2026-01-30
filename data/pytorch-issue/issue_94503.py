import torch
import textwrap
import io

tensors = [ 
    torch.randn(2, 2, dtype=torch.float),
    torch.randn(2, 2, dtype=torch.double),
    torch.randn(2, 2, dtype=torch.cfloat),
    torch.randn(2, 2, dtype=torch.cdouble),
    torch.randn(2, 2, dtype=torch.float16),
    torch.randint(-128, 128, [2, 2], dtype=torch.int8),
    torch.randint(-128, 128, [2, 2], dtype=torch.int16),
    torch.randint(-128, 128, [2, 2], dtype=torch.int32),
    torch.randint(-128, 128, [2, 2], dtype=torch.int64),
]

file = io.BytesIO()
torch.save(tensors, file)

file.seek(0)
data = file.read()
print("\n".join(textwrap.wrap(str(data), 80)))

with open('myfile.pt', 'wb') as f:
    f.write(file.getbuffer())

# Also save and print the BOM one as well