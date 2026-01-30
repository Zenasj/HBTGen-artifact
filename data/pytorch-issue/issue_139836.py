import torch

means = torch.randn(64, 3).cuda()
length_scales = torch.logspace(0.001, 0.1, 8).cuda()

@torch.compile
def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
    q_pos = means[q_idx]
    k_pos = means[k_idx]
    dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
    scale = length_scales[h]
    inv_dist = torch.exp(-dist / scale)
    return inv_dist * score

features = torch.randn(1, 8, 64, 64).cuda()
q = features
k = features
v = features

@torch.compile
def attn(q, k, v):
    return flex_attention(q, k, v, score_mod=euclidean_dist_pos_embed)

attn(q, k, v)

@torch.compile
def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
    q_pos = means[q_idx]
    k_pos = means[k_idx]

    #dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
    diff = q_pos - k_pos
    dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt()
    scale = length_scales[h]
    inv_dist = torch.exp(-dist / scale)
    return inv_dist * score