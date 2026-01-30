import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Timer, Compare
import torchinductor
from torchinductor.compile_fx import compile_fx_inner, cudagraphify
from torchinductor.decomposition import decompositions
from itertools import product
from functools import partial

torchinductor.config.debug = True

benchmark_name = "grid_sampler_2d"
Ns = [16]
Cs = [3]
# We will use same value for width and height
iHs = [128, 512, 1024]
oHs = [128, 512 , 1024]
# (interpolation_mode, padding_mode, align_corners)
options = [ (0, 0, False) , (0, 1, False), (0, 0, True), (0, 1, True) ]

def rand_uniform(*size, **kwargs):
    hi_val = kwargs['hi']
    lo_val = kwargs['lo']
    del kwargs['hi']
    del kwargs['lo']
    xs = torch.rand(*size, **kwargs)
    return (hi_val-lo_val)*xs + lo_val

def gen_inputs():
    make_arg = partial(rand_uniform, dtype=torch.float32, device="cuda", lo=-1.2, hi=1.2)
    for N, C, iH, oH,option in product(Ns, Cs, iHs, oHs, options):
        image_shape = (N, C, iH, iH)
        grid_shape = (N, oH, oH, 2)
        yield (make_arg(image_shape), make_arg(grid_shape), option)


def benchmark(label, sublabel, f, args):
    return Timer("f(*args)",
                 globals={"f": f, "args": args},
                 label=benchmark_name,
                 description=label,
                 sub_label=sublabel,
                 num_threads=torch.get_num_threads()).blocked_autorange()


def compare(image, grid, option):
    def f(image, grid):
        val = torch.ops.aten.grid_sampler_2d(image, grid, *option)
        return (val,)

    sublabel = f"{tuple(image.shape)}, {tuple(grid.shape)}, {tuple(map(int, option))}"
    print(sublabel)

    t_args = [image, grid]
    decomposed = make_fx(f, decomposition_table=decompositions, tracing_mode="fake")(*t_args)
    compiled_decomposed = compile_fx_inner(decomposed, t_args)
    yield benchmark("Decomposed", sublabel, compiled_decomposed, t_args)

    non_decomposed = make_fx(f, tracing_mode="fake")(*t_args)
    compiled_nondecomposed = compile_fx_inner(non_decomposed, t_args)
    yield benchmark("Lowering", sublabel, compiled_nondecomposed, t_args)

    # Just show the first two generated kernels
    if torchinductor.config.debug:
        torchinductor.config.debug = False

    cuda_f = cudagraphify(f, t_args)
    yield benchmark("Eager", sublabel, cuda_f, t_args)


if __name__ == '__main__':
    results = []
    for image, grid, option in gen_inputs():
        for res in compare(image, grid, option):
            results.append(res)

    compare = Compare(results)
    compare.trim_significant_figures()
    compare.print()