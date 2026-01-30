import torch

shapes = (1, 1)
dtype = torch.quint8

X = torch.rand(*shapes, dtype=torch.float) - 0.5
min_val = torch.min(X)
max_val = torch.max(X)
X_zero_point = int(torch.randint(-128, 127, (1,)))
num_bins = 2 ** 8
X_scale = float(max_val - min_val) / num_bins

c = X.shape[1]
mean = torch.rand(c).float()
var = torch.rand(c).float()
weight = torch.rand(c).float()
bias = torch.rand(c).float()
eps = 0.001

Y_zero_point = 1
Y_scale = 0.5

qx = torch.quantize_per_tensor(X, X_scale, X_zero_point, dtype)
qy = torch.ops.quantized.batch_norm1d_relu(qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)

assert qx.shape == qy.shape