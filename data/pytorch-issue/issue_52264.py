import torch, re, multiprocessing
from timeit import timeit
from torchbenchmark.models import LearningToPaint, pytorch_mobilenet_v3


def measure(fuse, benchmark_module, warmup=1, number=100):
    torch.set_num_threads(1)
    torch._C._jit_override_can_fuse_on_cpu(fuse)
    name = re.sub(r"^.*[.]", "", benchmark_module.__name__)
    benchmark = benchmark_module.Model(device="cpu", jit=True)
    model, example_inputs = benchmark.get_module()
    assert isinstance(model, torch.jit.ScriptModule)

    model = model.eval()
    timeit(lambda: model(*example_inputs), number=warmup)
    print(f"    script({name:20})         = {timeit(lambda: model(*example_inputs), number=number):.3f} sec")

    model = torch.jit.freeze(model)
    timeit(lambda: model(*example_inputs), number=warmup)
    print(f"    freeze(script({name:20})) = {timeit(lambda: model(*example_inputs), number=number):.3f} sec")


for fuse in (False, True):
    print(f"_jit_override_can_fuse_on_cpu({fuse}):")
    for benchmark_module in (LearningToPaint, pytorch_mobilenet_v3):
        # Doing a subproc to ensure we aren't running cached code
        p = multiprocessing.Process(target=measure, args=(fuse, benchmark_module))
        p.start()
        p.join()