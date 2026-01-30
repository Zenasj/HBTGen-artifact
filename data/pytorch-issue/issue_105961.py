import torch.nn as nn

import torch 
from torch.autograd import ProfilerConfig, ProfilerState, _disable_profiler, _enable_profiler
import pytest 
from typing import Any

class NsysProfilerScope:
    """
    Denotes the scope of code that nsys should profile when run with the --capture-range cudaProfilerApi and --capture-range-end stop command line flags.
    Users must ensure that this scope is entered and exited ONLY ONCE during program execution.
    Therefore, it is recommended that this scope be used at the highest level function that runs any PyTorch model.

    Usage:
        with NsysProfilerScope():
            <code to be profiled>
    """

    class_counter = 0  # Required to make sure multiple instances of the class are not created.

    def __init__(
        self,
        record_input_shape: bool = False, 
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental: bool = None, 
        is_unit_testing: bool = False,
    ):
        """
        Args:
            record_input_shape : bool flag to display shapes of tensor on nsys profile. Default = False
            profile_memory : bool flag to display tensor memory allocation / deallocation on nsys. Expensive to trace , hence , turned off by default
            with_stack : enable record source and information for the ops. Default = False
            with_flops : enable estimated FLOPS of operators. Default = False
            with_modules : record module hierarchy. Only supported for torchscript modules. Default = False
            is_unit_testing: helps in resetting class counter to perform unit testing on different configurations
        """
        super().__init__()

        ## Check only single instance of class is created
        if not is_unit_testing:
            NsysProfilerScope.class_counter += 1

        if NsysProfilerScope.class_counter > 1:
            raise RuntimeError("Multiple instances of NsysProfilerScope have been created ")

        self.profiler_config = ProfilerConfig(
            ProfilerState.NVTX,
            record_input_shape,
            profile_memory,
            with_stack,
            with_flops,
            with_modules,
            experimental,
        )

        self.activities = set()

        self.scopes = set()

        self.is_unit_testing = is_unit_testing

    def __enter__(self):
        _enable_profiler(self.profiler_config, self.activities, self.scopes)
        torch.cuda.synchronize()
        torch.cuda.profiler.cudart().cudaProfilerStart()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        torch.cuda.synchronize()
        torch.cuda.profiler.cudart().cudaProfilerStop()
        _disable_profiler()


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        u = t + t
        return u


@pytest.mark.parametrize("record_input_shape", [False])
@pytest.mark.parametrize("profile_memory", [False,True])
@pytest.mark.parametrize("with_stack", [False])
@pytest.mark.parametrize("with_flops", [False])
@pytest.mark.parametrize("with_modules", [False])
@pytest.mark.parametrize("experimental", [torch._C._profiler._ExperimentalConfig()])
@pytest.mark.parametrize("t_shape", [(3, 3, 3)])
@torch.no_grad()
def test_NsysProfilingScope(
    t_shape,
    record_input_shape,
    profile_memory,
    with_stack,
    with_flops,
    with_modules,
    experimental,
    is_unit_testing=True,
):
    t = torch.rand(t_shape).cuda()

    module = Module().eval().cuda()
    module = torch.jit.script(module)

    out = module(t)

    with NsysProfilerScope(
        record_input_shape, profile_memory, with_stack, with_flops, with_modules,experimental, is_unit_testing
    ):
        other = module(t)

    torch.allclose(out, other)

if __name__ == "__main__":
    pytest.main()