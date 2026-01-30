import torch

if (
                    not disable_meta
                    # TorchScript dumps a bunch of extra nonsense overloads
                    # which don't have corresponding dispatcher entries, we need
                    # to filter those out
                    and torch._C._dispatch_has_kernel(name)
                    # Don't register a meta kernel to any operator that has
                    # a CompositeImplicitAutograd kernel in core.
                    # Otherwise we won't be able to run autograd for that operator with the meta backend.
                    and "CompositeImplicitAutograd" not in torch._C._dispatch_dump(name)
                    and not torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")
                ):
                    meta_lib.impl(op_overload, fn)