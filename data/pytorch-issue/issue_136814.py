import torch

# We must wrap the whole thing into a function to reproduce the error.
def test():
    x = torch.ones(1)

    def fn():
        def inner():
            return x + 2
        return inner

    @torch.compile
    def start():
        fn_inner = fn()
        res = fn_inner()
        return res, fn_inner

    start()

test()

import torch

# The observed error comes from trying to resolve `InlinedClosureVariable` when
# binding args to a `NestedUserFunctionVariable` (so `x` and `inner` in the
# following scenario), ~~when Dynamo never built a representation of `x`.~~ Wrong,
# see https://github.com/pytorch/pytorch/blob/43a1d594ef051870b21eb28fc02b02359e5af640/torch/_dynamo/variables/functions.py#L228-L230
#
# ~~TLDR:Dynamo needs a way to represent captured variable that's defined outside
# the trace entry point.~~ Inaccurate, see below.
#
# TLDR: Dynamo needs a way to look up cell variable that's unmodified, and was
# captured in a frame that is no longer in the current tracer frames. See
# https://github.com/pytorch/pytorch/blob/43a1d594ef051870b21eb28fc02b02359e5af640/torch/_dynamo/variables/functions.py#L263-L276


# We must wrap the whole thing into a function to reproduce the error, otherwise
# inside `fn`, the captured variable `x` will be loaded as a global rather than
# closure variable, which bypass the whole `InlinedClosureVariable` thing.
def test():
    x = torch.ones([10])

    def fn():
        #   LOAD CLOSURE x
        #   BUILD TUPLE  1
        #   LOAD CONST   inner  ...code object...
        #   MAKE FUNCTION
        def inner():
            return x + 2
        return inner

    @torch.compile
    def start():
        fn_inner = fn()
        res = fn_inner()
        return res

    start()

test()