# torch.rand(5, 3, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        n = 512
        self.net = nn.Sequential(
            nn.Linear(3, n, bias=False),
            nn.LayerNorm(n),  # removing this fixes the issue
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n, 1)
        )
            
    def forward(self, x):
        return self.net(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(5, 3, dtype=torch.float32)

def calc_gradient_penalty(D, real_data, fake_data, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    data_shape = real_data.shape
    
    eps = torch.rand(data_shape[0], *([1] * (len(data_shape) - 1)))
    eps = eps.expand(data_shape)
    eps = eps.to(device)
    
    interpolates = eps * real_data + (1 - eps) * fake_data
    interpolates.requires_grad = True
    assert (interpolates.requires_grad)

    D_interpolates = D(interpolates)

    gradients = torch.autograd.grad(outputs=D_interpolates, 
                                    inputs=interpolates,
                                    grad_outputs=torch.ones_like(D_interpolates),
                                    create_graph=True,
                                    retain_graph=True, 
                                    only_inputs=True)[0]
    
    gradients = gradients.view(data_shape[0], -1)
    gradient_penalty = (gradients.norm(2, dim=1) - 1.0).pow(2).mean()
    return gradient_penalty

