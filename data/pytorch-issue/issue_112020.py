import torch.nn as nn

import torch
from torch import nn 

init_value = 0.4

class Spheres(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = torch.tensor([init_value], dtype=torch.float)
        self.register_buffer('radius', self.init_value, persistent=True)

def save(spheres, path):
    print('saving...')
    spheres.radius = torch.tensor([0.3])
    state_dict = {
            'spheres': spheres.state_dict(),
        }
    torch.save(state_dict, path)

    print('spheres.radius:', spheres.radius)         # tensor([0.3000])
    print('spheres.init_value:', spheres.init_value) # tensor([0.4000])

def load(spheres, path):
    print('loading...')
    state_dict = torch.load(path, map_location='cuda')
    spheres.load_state_dict(state_dict['spheres'])
    print('spheres.radius:', spheres.radius)         # tensor([0.3000])
    print('spheres.init_value:', spheres.init_value) # tensor([0.3000])

if __name__ == '__main__':
    spheres = Spheres(init_value)
    save(spheres, 'checkpoint_test.pkl')

    spheres2 = Spheres(init_value)
    load(spheres2, 'checkpoint_test.pkl')

import torch
from torch import nn

init_value = 0.4

class Spheres(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = torch.tensor([init_value], dtype=torch.float)
        self.register_buffer('radius', self.init_value, persistent=True)

def save(spheres, path):
    print('saving...')
    spheres.radius = torch.tensor([0.3])
    state_dict = {
            'spheres': spheres.state_dict(),
        }
    print(state_dict['spheres'])
    torch.save(state_dict, path)

    print('spheres.radius:', spheres.radius)         # tensor([0.3000])
    print('spheres.init_value:', spheres.init_value) # tensor([0.4000])

def load(spheres, path):
    print('loading...')
    print('spheres.radius:', spheres.radius)
    state_dict = torch.load(path, map_location='cuda')
    print(f"id(spheres.radius)={id(spheres.radius)}")
    print(f"id(spheres.init_value)={id(spheres.init_value)}")
    assert id(spheres.radius) == id(spheres.init_value)
    spheres.load_state_dict(state_dict['spheres'])
    print('spheres.radius:', spheres.radius)         # tensor([0.3000])
    print('spheres.init_value:', spheres.init_value) # tensor([0.3000])

if __name__ == '__main__':
    spheres = Spheres(init_value)
    save(spheres, 'checkpoint_test.pkl')

    spheres2 = Spheres(init_value)
    load(spheres2, 'checkpoint_test.pkl')