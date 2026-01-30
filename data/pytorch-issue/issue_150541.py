import torch
import timeit

def bench_mm(f, x, y):
    from torch.utils.benchmark import Timer
    return Timer(stmt="f(x, y); torch.mps.synchronize()",
                 globals={"x": x, "y": y, "f": f},
                  language="python", timer=timeit.default_timer).blocked_autorange()

x = torch.rand(1024, 512, device='mps')
y = torch.rand(512, 1, device='mps')

mm_c = torch.compile(torch.mm, options={"coordinate_descent_tuning": False})
mm_c_cdt = torch.compile(torch.mm, options={"coordinate_descent_tuning": True})

print(f"Compiled torch.mm perf (with cdt disabled) for 1024x512 and  512x1 matrices are {bench_mm(mm_c, x, y).median}")
print(f"Compiled torch.mm perf (with cdt enabled) for 1024x512 and  512x1 matrices are {bench_mm(mm_c_cdt, x, y).median}")