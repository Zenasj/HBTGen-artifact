@torch_op("aten::clone")
def aten_clone(
    self: TTensor, memory_format: str = ""  # pylint: disable=unused-argument
) -> TTensor:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    return op.Identity(self)