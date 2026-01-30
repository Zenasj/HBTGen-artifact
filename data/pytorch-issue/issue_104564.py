import torch
import math

mat = torch.matmul(t1_norm, t2_norm.T)

def cosine_similarity(t1, t2, dim=1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    # normalize, avoiding division by 0
    t1_norm = t1 / torch.clamp(t1_div, math.sqrt(eps))
    t2_norm = t2 / torch.clamp(t2_div, math.sqrt(eps))

    return (t1_norm * t2_norm).sum(dim=dim)

@torch.compile
def cosine_similarity(t1, t2, dim=-1, eps=1e-8):
    # get normalization value
    t1_div = torch.linalg.vector_norm(t1, dim=dim, keepdims=True)
    t2_div = torch.linalg.vector_norm(t2, dim=dim, keepdims=True)

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(math.sqrt(eps))
        t2_div.clamp_(math.sqrt(eps))

    # normalize, avoiding division by 0
    t1_norm = t1 / t1_div
    t2_norm = t2 / t2_div

    return (t1_norm * t2_norm).sum(dim=dim)