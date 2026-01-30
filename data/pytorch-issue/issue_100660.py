@torch_op("aten::mul")
def aten_mul(self: TReal, other: TReal) -> TReal:
    ...

@torch_op("aten::mul")
def aten_mul_bool(self: BOOL, other: BOOL) -> BOOL:
    ...

@torch_op("aten::argmax", trace_only=True)
def aten_argmax(
    self: TrealOrUInt8, dim: Optional[int] = None, keepdim: bool = False
) -> TrealOrUInt8:
    ...

@torch_op("aten::argmax", private=True)
def _aten_argmax_dim(self: TrealOrUInt8, dim: int, keepdim: bool = False) -> TrealOrUInt8:
    ...

@torch_op("aten::new_full")
def aten_new_full(self: TTensor, size: INT64, fill_value: TTensor) -> TTensor:
    ...

@torch_op("aten::new_full")
def aten_new_full_dtype(self: TTensor, size: INT64, fill_value: TTensor, dtype: int) -> TTensor:
    ...