import torch


class CtxMgr:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


@torch.compile(backend="eager", fullgraph=True)
def fn():
    with CtxMgr():
        with CtxMgr():
            pass
        with CtxMgr():
            with CtxMgr():
                pass
            torch._dynamo.graph_break()


fn()