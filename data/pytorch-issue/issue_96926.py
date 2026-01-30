import torch

python
#torch.onnx -> symbolic_opset9.py
@_onnx_symbolic("aten::bernoulli")
@_beartype.beartype
def bernoulli(g: jit_utils.GraphContext, input, generator=None, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )

    dtype = symbolic_helper._try_get_scalar_type(input)
    if dtype is None:
        return symbolic_helper._unimplemented(
            "Bernoulli", "input dtype not accessible", input
        )
    p = g.op(
        "RandomUniformLike",
        input,
        high_f=1.0,
        low_f=0.0,
        dtype_i=_type_utils.JitScalarType.from_name(dtype).onnx_type(),
    )
    output = g.op("Less", p, input)
    return g.op(
        "Cast", output, to_i=_type_utils.JitScalarType.from_name(dtype).onnx_type()
    )