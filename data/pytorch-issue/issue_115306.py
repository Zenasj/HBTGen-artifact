py
import torch
from torch.testing import make_tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils.benchmark import Timer, Compare
from torch._inductor.compile_fx import compile_fx_inner, cudagraphify_impl
from torch._inductor.decomposition import decompositions
from itertools import product
from functools import partial

torch._logging.set_logs(output_code=True)

benchmark_name = "put"
Ss = [512]


def gen_inputs():
    make_arg = partial(torch.randn, dtype=torch.float32, device="cuda")
    make_source = partial(torch.randn, dtype=torch.float32, device="cuda")

    def make_idx(n):
        return make_tensor((n,), device="cuda", dtype=torch.int64, low=0, high=n)

    for b, s, in product(Ss, Ss):
        yield make_arg((b * s)), make_idx(b), make_source(b)


def benchmark(label, f, x, idx, source):
    return Timer("f([x, idx, source])",
                 globals=locals(),
                 label=benchmark_name,
                 description=label,
                 sub_label=f"{tuple(x.shape)}",
                 num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=2)


def compare(x, idx, source):
    def f(args):
        x, idx, source = args
        val = torch.ops.aten.put(x, idx, source)
        return (val,)

    print(f"{tuple(x.shape)}")

    args = [x, idx, source]

    decomposed = make_fx(f, decomposition_table=decompositions, tracing_mode="fake")(args)
    compiled_decomposed = compile_fx_inner(decomposed, args, cudagraphs=False)
    yield benchmark("Decomposed", compiled_decomposed, *args)

    non_decomposed = make_fx(f, tracing_mode="fake")(args)
    compiled_nondecomposed = compile_fx_inner(non_decomposed, args, cudagraphs=False)
    yield benchmark("Lowering", compiled_nondecomposed, *args)

    # Just show the first two generated kernels
    torch._logging.set_logs(output_code=False)

    cuda_f = cudagraphify_impl(f, args, static_input_idxs=tuple(range(len(args))))
    yield benchmark("Eager", cuda_f, *args)


results = []
for args in gen_inputs():
    for res in compare(*args):
        results.append(res)

compare = Compare(results)
compare.trim_significant_figures()
compare.print()