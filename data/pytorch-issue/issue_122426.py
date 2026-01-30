import torch
import numpy as np

input = torch.tensor([np.inf], dtype=torch.float64)
part1 = torch.exp(input ** 2)  # tensor([inf], dtype=torch.float64)
part2 = torch.special.erfc(input)  # tensor([0.], dtype=torch.float64)

expected = part1 * part2  # expected result: tensor([nan], dtype=torch.float64)

actual = torch.special.erfcx(input)  # actual result: tensor([0.], dtype=torch.float64)