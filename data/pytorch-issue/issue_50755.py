import torch
import time

bug = False # Change to True to manifest a bug in rfactor

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C.te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

def rfac(loopnest, X, Y):
    loopnest.compute_inline(loopnest.get_loop_body_for(X))
    loops_y = loopnest.get_loops_for(Y)
    (o, i, t) = loopnest.split_with_tail(loops_y[2], 16)
    stmt = torch._C.te.simplify(loopnest.root_stmt())
    index = i.index_var()
    reduce_op = i.body().stmts()[0]
    if bug:
        loopnest.rfactor(reduce_op, index, o.body())
    else:
        loopnest.rfactor(reduce_op, index)
    stmt = torch._C.te.simplify(loopnest.root_stmt())
    loopnest.prepare_for_codegen()
    stmt = torch._C.te.simplify(loopnest.root_stmt())
    return stmt

with kernel_arena_scope():
    dtype = torch._C.te.Dtype.Float
    M = 256
    N = 256
    K = 256

    dM = torch._C.te.ExprHandle.int(M)
    dN = torch._C.te.ExprHandle.int(N)
    dK = torch._C.te.ExprHandle.int(K)

    dims_MN = [dM, dN]
    dims_NK = [torch._C.te.ExprHandle.int(i) for i in [N, K]]

    A = torch._C.te.Placeholder('A', dtype, [dM, dN])
    B = torch._C.te.Placeholder('B', dtype, [dN, dK])

    dim_args = [torch._C.te.DimArg(*args) for args in [(dM, 'm'), (dK, 'k'), (dN, 'n')]]

    def t(i, j):
        return B.load([j, i])
    tB = torch._C.te.Compute('tB', [dim_args[1], dim_args[2]], t)

    def compute(i, k, j):
        return A.load([i, j]) * tB.load([k, j])
    X = torch._C.te.Compute('X', dim_args, compute)

    Y = torch._C.te.SumReduce('Y', [dim_args[0], dim_args[1]], X, [dim_args[2]])

    loopnest = torch._C.te.LoopNest([Y])
    print(loopnest)

    stmt = rfac(loopnest, X, Y)

    tA = torch.rand(M, N) * 5
    tB = torch.rand(N, K) * 6
    tC = torch.empty(M, K)
    tR = torch.empty(M, K)

    cg = torch._C.te.construct_codegen('llvm', stmt, [torch._C.te.BufferArg(x) for x in [A, B, Y]])

    # warmup
    for _ in range(10):
        cg.call([tA, tB, tC])

    start = time.time()
    for _ in range(100):
        cg.call([tA, tB, tC])
    end = time.time()
    time1 = end - start

    # warmup
    for _ in range(10):
        torch.matmul(tA, tB, out=tR)

    start = time.time()
    for _ in range(100):
        torch.matmul(tA, tB, out=tR)
    end = time.time()
    time2 = end - start

    print('Results with NNC:')
    print('Time: ', time1)
    print('C[0][0] =', tC[0][0].item())
    print('Results of torch.matmul:')
    print('Time: ', time2)
    print('C[0][0] =', tR[0][0].item())