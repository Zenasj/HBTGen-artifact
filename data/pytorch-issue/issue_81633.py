import torch.nn as nn

import torch

class GAN(torch.nn.Module):
    def __init__(
                self,
                generator: torch.nn.Module,
                discriminator: torch.nn.Module) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def loss(self, real_data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.discriminator(real_data) - self.discriminator(self.generator(noise))

gan = GAN(generator=torch.nn.Linear(1, 2), discriminator=torch.nn.Linear(2, 1))

gradient_descent_ascent = torch.optim.SGD([
                {'params': gan.generator.parameters(), 'lr': 1.},
                {'params': gan.discriminator.parameters(), 'lr': -1.}
            ], lr=1.)

for real_data in train_loader:
    gradient_descent_ascent.zero_grad()
    noise = torch.randn(batch_size, 1)
    loss = gan.loss(real_data, noise)
    loss.backward()
    gradient_descent_ascent.step()

import torch

def my_sgd(p, g, lr, mom, mom_buff):
    if mom != 0:
        mom_buff = mom * mom_buff + (1 - mom) * g
        update = mom_buff
    else:
        update = g
    return p - lr * g, mom_buff

inp = torch.rand(2)
g = torch.rand(2)
mom_buff = torch.rand(2)

print("No momentum")
print(my_sgd(inp, g, -1., 0., None))
print(my_sgd(inp, -g, 1., 0., None))


print("With momentum")
print(my_sgd(inp, g, -1., 0.5, mom_buff))
print(my_sgd(inp, -g, 1., 0.5, mom_buff))