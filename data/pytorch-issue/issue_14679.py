import torch

# choose top K
candidate_weight, idx = torch.topk(weights, self.num_cxt, dim=1) # (batch, num_cxt), (batch, num_cxt)
idx_expand  = idx.unsqueeze(-1).expand(batch_size, self.num_cxt, self.fc7_dim) # (batch, num_cxt, fc7_dim)
# the error reported under the line
candidate_value = torch.gather(value, dim = 1, index = idx_expand) # (batch, topK, fc7_dim ) -> (batch, num_cxt, fc7_dim)