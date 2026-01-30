import numpy as np
import torch
x1 = np.ones((1, 32, 224, 224, 160))
ord = 2
print(np.size(x1))  # 256901120
res1 = torch.linalg.vector_norm(torch.tensor(x1, dtype=torch.float32), ord=ord)
res2 = torch.linalg.vector_norm(torch.tensor(x1, dtype=torch.float64), ord=ord)

print(res1, res2)  # tensor(11585.2373) tensor(16028.1353, dtype=torch.float64)
print(f"Expected result: {np.sqrt(np.size(x1))}")  # 16028.135262718493