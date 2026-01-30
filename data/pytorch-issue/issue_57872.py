import torch

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    i32 = torch._C._te.Dtype.Int
    N = torch._C._te.ExprHandle.int(10)
    start = torch._C._te.ExprHandle.int(0)
    stop = torch._C._te.ExprHandle.int(15) # intentional overflow
    i = torch._C._te.VarHandle('i', i32)

    X = torch._C._te.Placeholder('X', i32, [N])

    body = torch._C._te.Store.make(X.data(), [i], i)
    stmt = torch._C._te.For.make(i, start, stop, body)

    cg = torch._C._te.construct_codegen('ir_eval', stmt, [torch._C._te.BufferArg(X)])

    data = torch.zeros(20, dtype=torch.int32)
    cg.call([data])
    print(data)
    assert(data[10:].sum() == 0)