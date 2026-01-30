for i in 0..N:
  for j in 0..M:
    for k in 0..K:
      a[i,j,k] = 0

for t in 0..N*M*K:
  a[t/M/K, t/K%M, t%K] = 0

for t in 0..N*M*K:
  a[(t/M/K)*(M*K) + (t/K%M)*K + t%K] = 0

for t in 0..N*M*K:
  a[t] = 0

import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C.te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    dtype = torch._C.te.Dtype.Float
    M = 10
    K = 9
    N = 7

    def get_dim_args(dims):
        dim_args = []
        for dim in dims:
            dim_args.append(torch._C.te.DimArg(dim, 'i' + str(len(dim_args))))
        return dim_args

    (MM, KK, NN) = [torch._C.te.ExprHandle.int(x) for x in [M, K, N]]

    dtype = torch._C.te.Dtype.Float
    X = torch._C.te.Placeholder('X', dtype, [MM, KK, NN])
    Y = torch._C.te.Compute('T', get_dim_args([MM, KK, NN]),
            lambda m, n, k: X.load([m, n, k]))
    loopnest = torch._C.te.LoopNest([Y])
    loops = loopnest.get_loops_for(Y)
    loopnest.flatten(loops)
    loopnest.prepare_for_codegen()
    loopnest.simplify()
    s = loopnest.root_stmt()
    print(s)