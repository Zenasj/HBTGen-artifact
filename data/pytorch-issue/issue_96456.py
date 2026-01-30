import torch

x = torch.rand((1, 12, 256*64), requires_grad=True)

def transpose_for_scores(x):
    new_x_shape = x.size()[:-1] + (256, -1)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def fn(x):
    scale_factor = 0.5
    x = x.relu()
    x = transpose_for_scores(x)
    x /= torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float) * scale_factor)
    return x.transpose(-1, -2)

fn(x)
torch.compile(fn)(x)