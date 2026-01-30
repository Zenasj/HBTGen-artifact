import torch

torch.manual_seed(42)
noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)