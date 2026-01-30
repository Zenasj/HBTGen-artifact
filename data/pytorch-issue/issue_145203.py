import torch

# On CPU
zeros = torch.zeros((10, 2), dtype=torch.int16, device="cpu")
zeros[:, 0] ^= 1
print(zeros)  # Expected and correct output:
# tensor([[1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0]], dtype=torch.int16)

# On MPS
zeros = torch.zeros((10, 2), dtype=torch.int16, device="mps")
zeros[:, 0] ^= 1
print(zeros)  # Incorrect output:
# tensor([[1, 1],
#         [1, 1],
#         [1, 1],
#         [1, 1],
#         [1, 1],
#         [0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0]], device='mps:0', dtype=torch.int16)

# Non-in-place workaround
zeros = torch.zeros((10, 2), dtype=torch.int16, device="mps")
zeros[:, 0] = zeros[:, 0] ^ 1
print(zeros)  # Correct output:
# tensor([[1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0]], device='mps:0', dtype=torch.int16)