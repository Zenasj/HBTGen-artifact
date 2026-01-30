import torch.nn.functional as F

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import pickle as pkl

torch.manual_seed(1)


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

torch.backends.cudnn.enabled = False               
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

decoder = Decoder(3, 32).to(device)
decoder.train()
optimizer = optim.Adam(decoder.parameters())

with open('minibatch.pkl', 'rb') as f:
    data = pkl.load(f)

for batch_idx in range(500):
    data = data.to(device)
    mu = torch.randn((32, 32), device=device)
    recon_batch = decoder(mu)
    loss = .5 * (recon_batch - data).pow(2).sum(dim=(1,2,3)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if batch_idx % 20 == 0:
        print(f'Loss: {loss.item()/len(data):.6f}')