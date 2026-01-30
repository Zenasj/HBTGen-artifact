3
import torch

print("Single element comparison")

complex_tensor = torch.complex(
    torch.arange(10, dtype=torch.float32),
    torch.arange(10, dtype=torch.float32),
)

for i in range(complex_tensor.size(0)):
    if complex_tensor.angle()[i] != complex_tensor[i].angle():
        print("Mismatch at pos", i)
# Output:
# Mismatch at pos 1
# Mismatch at pos 2
# Mismatch at pos 3
# Mismatch at pos 4
# Mismatch at pos 5
# Mismatch at pos 6
# Mismatch at pos 7

print()

########################################################

print("Concatenate before and after `angle()`")

complex_tensor_1 = torch.complex(
    torch.arange(0, 5, dtype=torch.float32),
    torch.arange(0, 5, dtype=torch.float32),
)
complex_tensor_2 = torch.complex(
    torch.arange(5, 10, dtype=torch.float32),
    torch.arange(5, 10, dtype=torch.float32),
)

concat_before_angle = torch.cat([complex_tensor_1, complex_tensor_2], dim=0).angle()
concat_after_angle = torch.cat([complex_tensor_1.angle(), complex_tensor_2.angle()], dim=0)

print("Results are identical:", (concat_before_angle == concat_after_angle).all().item())
# Output:
# Results are identical: False