import torch.nn as nn

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
    #     run_tests(needs="filelock")

    CompiledOptimizerTests()

def make_test(optim_cls, closure=None, kernel_count=2, **kwargs):
    @requires_cuda()
    def test_fn(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        input = torch.ones([10, 10], device="cuda:0")
        model_eager = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device="cuda:0") for _ in range(2)]
        )
        model_eager(input).sum().backward()

        input = torch.ones([10, 10], device="cuda:0")
        model_compiled = deepcopy(model_eager)
        model_compiled(input).sum().backward()

        opt_eager = optim_cls(model_eager.parameters(), **kwargs)
        opt_compiled = optim_cls(model_compiled.parameters(), **kwargs)
        compiled_step = compile_opt(opt_compiled, closure=closure)

        with torch.set_grad_enabled(False):
            compiled_step()
            opt_eager.step()

        self.assertEqual(
            list(model_eager.parameters()), list(model_compiled.parameters())
        )

        if self.check_kernel_count:
            # currently, we compile the step and the rest of the computation
            # separately because the step is a single element tensor
            # hence, the usual kernel count is 2
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, kernel_count
            )

    return test_fn

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
...
...
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")

import torch
print(torch.cuda.is_available())