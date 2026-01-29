# torch.rand(1, 1, H, W, dtype=torch.complex128)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
from torch.fft import fft2, ifft2, fftshift

class MyModel(nn.Module):
    def __init__(self, delta, wavelength, distance):
        super(MyModel, self).__init__()
        self.delta = delta
        self.wavelength = wavelength
        self.distance = distance

    def forward(self, target):
        # Phase
        target = torch.ones_like(target) * torch.exp(2j * torch.pi * target)
        
        # Index X
        indexX = torch.tensor([[i for i in range(target.shape[-1])] for _ in range(target.shape[-2])], dtype=torch.double, device=target.device)
        
        # Index Y
        indexY = torch.tensor([[i for _ in range(target.shape[-1])] for i in range(target.shape[-2])], dtype=torch.double, device=target.device)
        
        # Coordinate X
        coordX = indexX - target.shape[-1] / 2 + 0.5
        
        # Coordinate Y
        coordY = indexY - target.shape[-2] / 2 + 0.5
        
        # Fx
        diff_X = coordX / (self.delta[1] * target.shape[-1])
        
        # Fy
        diff_Y = coordY / (self.delta[0] * target.shape[-2])
        
        # Phase
        phi = 2 * torch.pi * self.distance / self.wavelength * torch.sqrt(1 - (self.wavelength ** 2) * (diff_X ** 2 + diff_Y ** 2))
        
        # Window
        window = torch.conj(torch.exp(1j * phi))
        
        # Propagation #1
        middle = fftshift(fft2(fftshift(target)))
        
        # Propagation #2
        middle = middle * window
        
        # Propagation #3
        result = fftshift(ifft2(fftshift(middle)))
        
        # Result
        result = torch.abs(result) / torch.max(torch.abs(result))
        
        return result

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    delta = (8.0e-6, 8.0e-6)
    wavelength = 515e-9
    distance = 0.1
    return MyModel(delta, wavelength, distance)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    H, W = 256, 256  # Example dimensions
    target = torch.rand(1, 1, H, W, dtype=torch.double, device='cuda:0')
    return target

