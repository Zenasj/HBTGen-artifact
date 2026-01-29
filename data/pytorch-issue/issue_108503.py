# torch.rand(1, 2, 600, 80, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = x.cuda().bfloat16()  # Convert to CUDA and bfloat16 as in the example

        # Split into four sub_sequences along the sequence dimension (third dimension)
        sub1 = x[:, :, :128, :]
        sub2 = x[:, :, 128:256, :]
        sub3 = x[:, :, 256:384, :]
        sub4 = x[:, :, 543:, :]

        # Compute individual attentions without scale parameter (replicates the bug scenario)
        att1 = F.scaled_dot_product_attention(sub1, sub1, sub1)
        att2 = F.scaled_dot_product_attention(sub2, sub2, sub2)
        att3 = F.scaled_dot_product_attention(sub3, sub3, sub3)
        att4 = F.scaled_dot_product_attention(sub4, sub4, sub4)

        # Create mask to enforce self-attention within each sub-sequence
        B, N, S, E = x.shape
        mask = torch.zeros(B, N, S, S, device=x.device, dtype=torch.bool)
        mask[:, :, :128, :128] = True
        mask[:, :, 128:256, 128:256] = True
        mask[:, :, 256:384, 256:384] = True
        mask[:, :, 543:, 543:] = True

        # Compute full attention with mask (without scale parameter)
        full_att = F.scaled_dot_product_attention(x, x, x, attn_mask=mask)

        # Extract corresponding parts from full_att
        part1 = full_att[:, :, :128, :]
        part2 = full_att[:, :, 128:256, :]
        part3 = full_att[:, :, 256:384, :]
        part4 = full_att[:, :, 543:, :]

        # Compare using allclose with atol=1e-6
        close1 = torch.allclose(att1, part1, atol=1e-6)
        close2 = torch.allclose(att2, part2, atol=1e-6)
        close3 = torch.allclose(att3, part3, atol=1e-6)
        close4 = torch.allclose(att4, part4, atol=1e-6)

        # Return True only if all comparisons are close (should return False due to the bug)
        return torch.all(torch.tensor([close1, close2, close3, close4], device=x.device))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 600, 80, dtype=torch.float32)

