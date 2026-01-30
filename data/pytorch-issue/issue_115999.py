import torch.nn as nn

def compile_models(generator, discriminator, gan, latent_dim):
     d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
     g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

     criterion = nn.BCELoss()

     discriminator.compile(optimizer=d_optimizer, loss=criterion)
     gan.compile(optimizer=g_optimizer, loss=criterion)