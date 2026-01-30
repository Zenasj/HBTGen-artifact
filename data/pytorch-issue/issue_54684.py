import torch

n_frames = speechs_hat.shape[-1]
r_ss_loc = torch.einsum(
    "nift, njft -> nfij", speechs_hat, speechs_hat.conj() / n_frames
)