import torch
@torch.jit.script
def rotate_points_export(points):
    return points

def xxx(input):
    outputs = {}
    outputs['a'] = input
    outputs['b'] = input

    outputs['b'] = rotate_points_export(outputs['b'])
    return outputs

points = torch.rand((1,2,3,4))

model = torch.jit.trace(xxx, points, strict=False)
torch.jit.save(model, 'xx.pt')