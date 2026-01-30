import torch

_, asym_id_counts = torch.unique(
    gt_features["asym_id"], sorted=True, return_counts=True, dim=-1
)