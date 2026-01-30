import torch
import torch.fx.experimental
import torch.fx.experimental.proxy_tensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

a = torch.tensor([1.0, 2.0, 3.0])
shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env)
fake_a = fake_mode.from_tensor(a)


def run_mul(non_fallthrough_keys: torch.DispatchKeySet, *, has_fallthrough: bool) -> torch.Tensor:
    with (
        fake_mode,
        torch._C._EnablePythonDispatcher(),
        torch._C._IncludeDispatchKeyGuard(torch.DispatchKey.Meta),
        torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch.DispatchKey.Python).add(
                torch.DispatchKey.PythonTLSSnapshot
            )
        ),
    ):
        fake_mode.in_kernel_invocation = True
        expected_key_set = torch._ops.key_extractor(tensors=[fake_a], key_mask=non_fallthrough_keys)

        assert expected_key_set.highestPriorityTypeId() == torch.DispatchKey.PythonDispatcher
        has_meta = expected_key_set.has(torch.DispatchKey.Meta)
        if has_fallthrough:
            assert not has_meta
        else:
            assert has_meta

        # Call mul operator
        return 5 * fake_a


if __name__ == "__main__":
    lib = torch.library.Library("aten", "IMPL", "CPU")
    non_fallthrough_keys = torch._C._dispatch_keyset_full()

    # No error
    run_mul(non_fallthrough_keys, has_fallthrough=False)

    # makeFallthrough on CPU will remove all dense keys (likewise for any other dense key):
    #   CPU, CUDA, HIP, XLA, MPS, IPU, XPU, HPU, VE, Lazy, MTIA, PrivateUse1, PrivateUse2, PrivateUse3, Meta,
    lib.impl(torch.ops.aten.mul.Tensor, torch.library.fallthrough_kernel)
    non_fallthrough_keys = non_fallthrough_keys.remove(torch.DispatchKey.CPU)
    # Does not contain CPU or Meta
    assert not non_fallthrough_keys.has(torch.DispatchKey.CPU) and not non_fallthrough_keys.has(
        torch.DispatchKey.Meta
    )

    # RuntimeError: TensorIterator does not support symbolic shapes; please implement this operator in torch/_refs using the elementwise or reduction helpers (look at backtrace to find out what operator this is)
    run_mul(non_fallthrough_keys, has_fallthrough=True)