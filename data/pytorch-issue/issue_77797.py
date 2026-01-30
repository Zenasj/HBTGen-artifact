import torch

latent = 2 * torch.randn([1, self.z_dim]).to(self.device)
label = None
image = generator(latent, label, truncation_psi=1, noise_mode='const')
image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]