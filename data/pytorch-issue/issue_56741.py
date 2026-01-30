simplified_stmts = loopnest.simplify().stmts()
stmt = simplified_stmts[-1]
print(f"Original stmt:\n{stmt}")
print(f"Before vectorization:\n{loopnest.root_stmt()}")
stmt = loopnest.flatten([stmt])
stmt = loopnest.normalize(stmt)
stmt = loopnest.get_innermost_loops_for(Pconv.data())[-1]
loopnest.vectorize(stmt)
print(f"After vectorization:\n{loopnest.root_stmt()}")

import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

def repro(vectorize):
    print ("********************************")
    print ("Vectorize = ", vectorize)
    with kernel_arena_scope():
        dtype = torch._C._te.Dtype.Float
        N = 13
        M = 2

        dn = torch._C._te.ExprHandle.int(N)
        dm = torch._C._te.ExprHandle.int(M)

        X = torch._C._te.Placeholder('X', dtype, [dm, dn])
        Y = torch._C._te.Placeholder('Y', dtype, [dm, dn])

        dim_args = [torch._C._te.DimArg(*args) for args in [(dm, 'm'), (dn, 'n')]]

        def compute(m, n):
            return X.load([m, n]) + Y.load([m, n])
        Z = torch._C._te.Compute('Z', dim_args, compute)
        loopnest = torch._C._te.LoopNest([Z])
        print("original", loopnest)
        loops = loopnest.get_loops_for(Z)
        loopnest.slice_head(loops[1], 1)
        loopnest.simplify()
        print("after peel", loopnest)
        loops = loopnest.get_innermost_loops_for(Z.buf())
        if vectorize:
            loopnest.vectorize(loops[1])
            loopnest.simplify()
            print("after vectorize", loopnest)
        else:
            print("Skip vectorization")

        tX = torch.arange(0, N * 3, dtype=torch.float).reshape(3, N) + 1.
        tY = torch.arange(0, N * 3, dtype=torch.float).reshape(3, N) + 2.
        tZ = torch.empty(2, N)

        codegen = torch._C._te.construct_codegen('ir_eval', loopnest.root_stmt(), [torch._C._te.BufferArg(x) for x in [X, Y, Z]])
        codegen.call([tX, tY, tZ])
        print(tZ)
        assert(torch.allclose(tZ, (tX + tY)[0:2, 0:N]))
        print("Success!")

repro(vectorize=False) # Succeeds
repro(vectorize=True)  # Fails