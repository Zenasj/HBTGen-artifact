import torch
width = 12*12
val = [float(1) for i in range(0,width)]
row = col = [j for j in range(0,width)]
indices = [row, col]
L_matrix = grad_matrix = torch.sparse_coo_tensor(indices, val, (width, width), dtype=torch.float64).coalesce()

row_L, col_L = L_matrix.indices()