# torch.rand(20, 32, 32, 32, 3, 3, dtype=torch.float)

import torch
import torch.nn as nn

class BatchMatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch):
        assert batch.dtype == torch.float
        batch_shape = batch.shape
        mat_size = batch_shape[-1]
        batch = batch.reshape(-1, mat_size, mat_size)
        theta_thres = 1.461661507209034e+00
        norms = batch.data.abs().sum(-2).max(-1).values
        s = (torch.floor(torch.log2(norms / theta_thres)) + 1).clamp(min=0)
        s_max = s.max().cpu()
        has_nan = torch.isnan(s_max).item()
        s_max = s_max.item()
        if has_nan:
            return torch.full_like(batch, torch.nan)
        output_scaled = torch.matrix_exp(
            batch * torch.pow(2.0, -s).reshape(-1, 1, 1))
        sorted_s, sorted_s_inds = torch.sort(s, dim=0)
        sorted_s_aug = torch.nn.functional.pad(sorted_s, (0, 1), value=s_max + 1)
        split_edges = (torch.diff(sorted_s_aug) > 0).nonzero()[:, 0]
        pows = torch.pow(2, sorted_s[split_edges].long())
        split_sizes = torch.nn.functional.pad(split_edges, (1, 0), value=-1)
        split_sizes = torch.diff(split_sizes)
        split_sizes = tuple(split_sizes.cpu().numpy().tolist())
        ind_pieces = torch.split(sorted_s_inds, split_sizes, dim=0)
        output_pieces = []
        for pow_v, ind_piece in zip(pows, ind_pieces):
            output_pieces.append(torch.matrix_power(
                output_scaled[ind_piece], pow_v))
        output = torch.cat(output_pieces, dim=0)
        output = output[torch.argsort(sorted_s_inds)]
        output = output.reshape(batch_shape)
        return output

    @staticmethod
    def backward(ctx, grad):
        batch, = ctx.saved_tensors
        dim = batch.shape[-1]
        batch_T = torch.transpose(batch, -2, -1)
        larger_mat = torch.cat((
            torch.cat((batch_T, grad), dim=-1),
            torch.cat((torch.zeros_like(grad), batch_T), dim=-1)), dim=-2)
        adjoint_result = BatchMatrixExp.apply(larger_mat)
        return torch.narrow(
            torch.narrow(adjoint_result, -2, 0, dim),
            -1, dim, dim)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return BatchMatrixExp.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(20, 32, 32, 32, 3, 3, dtype=torch.float)

