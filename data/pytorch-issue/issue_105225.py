import torch.nn as nn

import time
import torch
ti = torch.tensor([[[[[[ 1.3928, -0.6127,  0.1813],
            [-0.7089,  1.6513, -0.4091],
            [-0.0032, -0.1018,  1.1634]]]]]])



for f in [0.1,1,2]:
    for dev in ["cpu","cuda"]:
        tmp2 = torch.zeros([20,32,32,32,3,3]).to(dev)
        tmp2[...] = f*ti[None,None,None,None,:,:].to(dev)
        
        start = time.time()
        tmp = torch.matrix_exp(tmp2)
        print("dev:",dev," time:",time.time()-start," f:",f)

import torch
from torch.utils.benchmark import Compare, Timer


def get_timer(A):
    timer = Timer(
        r""" torch.matrix_exp(A)""",
        globals=locals(),
        label="matrix_exp",
        description=f"{A.device}",
        sub_label=f"shape {tuple(A.shape)}",
        num_threads=torch.get_num_threads()
    )
    return timer.blocked_autorange()


def get_params():
    shape = [20,32,32,32,3,3]
    for device in ("cpu", "cuda"):
        yield (torch.randn(shape, device=device),)


compare = Compare([get_timer(*params) for params in get_params()])
compare.trim_significant_figures()
compare.print()

3
class BatchMatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(batch):
        """Workaround for poor matrix_exp parallelism for large batches
        See https://github.com/pytorch/pytorch/issues/107291 for details
        The result may be less precise than torch.matrix_exp"""
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
            # As of PyTorch 2.0.1, matrix_exp on nan causes undefined behavior
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
        output = torch.reshape(output, batch_shape)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        batch, = ctx.saved_tensors
        dim = batch.shape[-1]
        batch_T = torch.transpose(batch, -2, -1)
        larger_mat = torch.cat((
            torch.cat((batch_T, grad), dim=-1),
            torch.cat((torch.zeros_like(grad), batch_T), dim=-1)), dim=-2)
        return torch.narrow(torch.narrow(
            BatchMatrixExp.apply(larger_mat), -2, 0, dim), -1, dim, dim)

import torch
from torch.utils.benchmark import Timer, Compare


def mexp(A):
    """Workaround for poor matrix_exp parallelism for large batches
    See https://github.com/pytorch/pytorch/issues/107291 for details
    The result may be less precise than torch.matrix_exp"""
    A_shape = A.shape
    A = A.flatten(end_dim=-3)
    theta_thres = 1.461661507209034e+00
    norms = torch.linalg.matrix_norm(A, ord=1)
    s = (torch.floor(torch.log2(norms / theta_thres)) + 1).clamp(min=0)
    s_max = s.max().cpu().item()
    is_nan = s_max != s_max
    if is_nan:
        # As of PyTorch 2.0.1, matrix_exp on nan causes undefined behavior
        return torch.full_like(A, torch.nan)
    # rescale
    output_scaled = torch.matrix_exp(A * torch.pow(2.0, -s).view(-1, 1, 1))

    # sort
    sorted_s, sorted_s_inds = torch.sort(s, dim=0)
    split_counts = torch.unique_consecutive(sorted_s, return_counts=True)[1]
    split_edges = torch.cumsum(split_counts, dim=0) - 1
    split_adv = torch.diff(split_edges, prepend=split_edges.new_zeros([1]))
    unique_s = sorted_s[split_edges].long()
    diffs = torch.diff(unique_s, prepend=unique_s.new_zeros([1]))

    idxs = split_adv.tolist()
    ps = diffs.tolist()

    acc = output_scaled[sorted_s_inds]
    output_pieces = []
    for i, p in zip(idxs, ps):
        for _ in range(p):
            acc = acc @ acc
        output_pieces.append(acc[:i+1])
        acc = acc[i+1:]

    # Compose the result back
    output = torch.cat(output_pieces, dim=0)
    output = output[torch.argsort(sorted_s_inds)]
    output = torch.reshape(output, A_shape)
    return output


def gen_inputs():
    x = torch.randn(1000000, 8, 8, device="cuda")
    # 174ms
    for fn in (mexp,):
        yield fn, x


def benchmark(f, x):
    torch.testing.assert_close(f(x[:8]), torch.matrix_exp(x[:8]))
    return Timer("f(x)",
                 globals=locals(),
                 label="Norm",
                 sub_label=f"{f.__name__}",
                 description=str(tuple(x.shape)),
                 num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=2)


results = []
for args in gen_inputs():
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()