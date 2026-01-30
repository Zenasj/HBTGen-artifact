import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    dtype = torch._C._te.Dtype.Float
    M = 1
    K = 9
    N = 1

    def get_dim_args(dims):
        dim_args = []
        for dim in dims:
            dim_args.append(torch._C._te.DimArg(dim, 'i' + str(len(dim_args))))
        return dim_args

    (MM, KK, NN) = [torch._C._te.ExprHandle.int(x) for x in [M, K, N]]

    dtype = torch._C._te.Dtype.Float
    X = torch._C._te.Placeholder('X', dtype, [MM, KK, NN])
    Y = torch._C._te.Compute('Y', get_dim_args([MM, KK, NN]),
            lambda m, n, k: X.load([m, n, k]))
    Z = torch._C._te.Compute('Z', get_dim_args([MM, KK, NN]),
            lambda m, n, k: Y.load([m, n, k]))
    loopnest = torch._C._te.LoopNest([Z])
    print('Original statement:', loopnest.root_stmt())

    # If we un-comment this 'simplify', compute_inline will fail
    # loopnest.simplify()
    # print('After simplify:', loopnest.root_stmt())

    loopnest.compute_inline(loopnest.get_loop_body_for(Y))
    print('After compute_inline(Y):', loopnest.root_stmt())