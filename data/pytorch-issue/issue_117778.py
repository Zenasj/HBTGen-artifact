import torch

#torch.set_printoptions(threshold=10000)

def roll_and_zero_first_row(x):
    x = torch.roll(x, -1, 0)
    x[0].fill_(0.0)
    return x

size = (1024, 2)

inp = torch.ones(size, device="cuda")
compiled = torch.compile(roll_and_zero_first_row)
out_eager = roll_and_zero_first_row(inp)
out_compiled = compiled(inp)

print((out_eager - out_compiled).abs())
print(out_eager)
print(torch.testing.assert_allclose(out_eager, out_compiled))