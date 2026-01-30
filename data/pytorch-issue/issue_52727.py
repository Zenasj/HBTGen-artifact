import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    dtype = torch._C._te.Dtype.Float
    K = 100
    KK = torch._C._te.ExprHandle.int(K)

    B = torch._C._te.Compute('buf', [torch._C._te.DimArg(KK, 'i')], lambda k: k*k+k)
    C = torch._C._te.Compute('buf', [torch._C._te.DimArg(KK, 'i')],
            lambda k: B.load([k])*B.load([k]) - B.load([k]))

    loopnest = torch._C._te.LoopNest([B, C])
    print('Aggregated stmt:', loopnest.root_stmt())
    loops_B = loopnest.get_loops_for(B)
    loops_C = loopnest.get_loops_for(C)
    print('First loop:', loops_B[0])
    print('Second loop:', loops_C[0])