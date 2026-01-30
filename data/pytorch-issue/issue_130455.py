import torch


def normalize_points(points_3d):
    return points_3d[:, :2] / points_3d[:, 2:3]


# Test on CPU
device = "cpu"
points_3d = torch.rand(100, 3, device=device) 

compiled_normalize = torch.compile(normalize_points)
result = compiled_normalize(points_3d)
print(f"Compilation successful on {device.upper()}")

# Test on GPU
device = "cuda"
points_3d = torch.rand(100, 3, device=device) 

compiled_normalize = torch.compile(normalize_points)
result = compiled_normalize(points_3d)
print(f"Compilation successful on {device.upper()}")