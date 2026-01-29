# torch.rand(7000,3, device='cuda'), torch.rand(7000,3, device='cuda'), torch.rand(7000,7000, device='cuda')

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        src_points, ref_points, log_n_affinity = inputs
        
        ref_dist = torch.cdist(ref_points, ref_points)
        src_dist = torch.cdist(src_points, src_points)
        
        ref_ne_idx = ref_dist.topk(6, dim=-1, largest=False)[1][:,1:]  # 5+1 → top 6, then drop first
        src_ne_idx = src_dist.topk(6, dim=-1, largest=False)[1][:,1:]
        
        src_ne_dist = torch.linalg.norm(
            src_points[src_ne_idx] - src_points.unsqueeze(1).repeat(1,5,1), dim=-1
        )
        ref_ne_dist = torch.linalg.norm(
            ref_points[ref_ne_idx] - ref_points.unsqueeze(1).repeat(1,5,1), dim=-1
        )
        
        n_s, n_r = log_n_affinity.size()
        pair_affinity = torch.zeros((n_s, n_r, 5,5), device=src_points.device)
        
        for i in range(5):
            for j in range(5):
                temp_dist = src_ne_dist[:,i].unsqueeze(-1).repeat(1, n_r) - ref_ne_dist[:,j].unsqueeze(0).repeat(n_s,1)
                temp_dist = temp_dist ** 2 / 0.1 ** 2
                temp_dist = torch.clamp(1 - temp_dist, min=0.)
                pair_affinity[:,:,i,j] = log_n_affinity * temp_dist * log_n_affinity[src_ne_idx[:,i]][:, ref_ne_idx[:,j]]
        
        _, indices = torch.topk(pair_affinity.view(-1), 512, dim=-1)
        
        first_node_src = (indices // (n_r * 5 * 5)).long()
        first_node_ref = ((indices % (n_r * 5 * 5)) // (5 * 5)).long()
        second_idx = (indices % (n_r * 5 * 5)) % (5 * 5)
        second_node_src = (second_idx // 5).long()
        second_node_ref = second_idx % 5
        
        second_node_src = src_ne_idx[first_node_src, second_node_src]
        second_node_ref = ref_ne_idx[first_node_ref, second_node_ref]
        
        return first_node_src, first_node_ref, second_node_src, second_node_ref

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device('cuda')
    src_points = torch.randn(7000, 3, device=device)
    ref_points = torch.randn(7000, 3, device=device)
    log_n_affinity = torch.randn(7000, 7000, device=device)
    return (src_points, ref_points, log_n_affinity)

