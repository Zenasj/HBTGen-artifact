import onnxscript
import torch

# Assuming you use opset18
from onnxscript.onnx_opset import opset18 as op

custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=18)


@onnxscript.script(custom_opset)
def ScaledDotProductAttention(
    query,
    key,
    value,
    dropout_p,
):
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    embedding_size = op.CastLike(op.Shape(query)[-1], query)
    scale = op.Div(1.0, op.Sqrt(embedding_size))

    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def custom_scaled_dot_product_attention(g, query, key, value, attn_mask, dropout, is_causal, scale=None):
    return g.onnxscript_op(ScaledDotProductAttention, query, key, value, dropout).setType(query.type())

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::scaled_dot_product_attention",
    symbolic_fn=custom_scaled_dot_product_attention,
    opset_version=18,
)

# Custom operation registered, we can produce our onnx model

from dp.model.model import load_checkpoint
from dp.phonemizer import Phonemizer
from dp.preprocessing.text import Preprocessor

model, checkpoint = load_checkpoint('latin_ipa_forward.pt')

dummy_input = {"batch": {"text":torch.rand((32,32)).long()}}

torch.onnx.export(
    model,                      # PyTorch Model
    dummy_input,                # Input tensor
    "latin_ipa_forward.onnx",   # Output file (eg. 'output_model.onnx')
    custom_opsets = {"torch.onnx": 18},           
    opset_version=18,           # Operator support version
    input_names=['embedding'],  # Input tensor name (arbitary)
    output_names=['fc_out']     # Output tensor name (arbitary)
)

@onnxscript.script(custom_opset)
def aten_unflatten(self: TReal, dim: INT64, sizes: INT64):
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

    self_size = op.Shape(self)

    if dim < 0:
        # PyTorch accepts negative dim as reversed counting
        self_rank = op.Size(self_size)
        dim = self_rank + dim

    head_start_idx = op.Constant(value_ints=[0])
    head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
    head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

    tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    tail_end_idx = op.Constant(value_ints=[_INT64_MAX])
    tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

    final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

    return op.Reshape(self, final_shape)

# Registering custom operation for unflatten
@onnxscript.script(custom_opset)
def aten_unflatten(self, dim, sizes):
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

    self_size = op.Shape(self)

    if dim < 0:
        # PyTorch accepts negative dim as reversed counting
        self_rank = op.Size(self_size)
        dim = self_rank + dim

    head_start_idx = op.Constant(value_ints=[0])
    head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
    head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

    tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    #tail_end_idx = op.Constant(value_ints=[_INT64_MAX])
    tail_end_idx = op.Constant(value_ints=[9223372036854775807]) # = sys.maxint, exactly 2^63 - 1 -> 64 bit int
    tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

    final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

    return op.Reshape(self, final_shape)

def custom_unflatten(g, *dim, **shape):
    return g.onnxscript_op(aten_unflatten, *dim, **shape)    

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::unflatten",
    symbolic_fn=custom_unflatten,
    opset_version=18,
)

def custom_unflatten(g, self, dim, shape):
    return g.onnxscript_op(aten_unflatten, self, dim, shape).setType(self.type().with_sizes([1, 2, 3, 4, 5]))  # For example, you know size would be 1,2,3,4,5

custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=18)

@onnxscript.script(custom_opset)
def aten_unflatten(self, dim, sizes):
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

    self_size = op.Shape(self)

    if dim < 0:
        # PyTorch accepts negative dim as reversed counting
        self_rank = op.Size(self_size)
        dim = self_rank + dim

    head_start_idx = op.Constant(value_ints=[0])
    head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
    head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

    tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    tail_end_idx = op.Constant(value_ints=[9223372036854775807]) # = sys.maxint, exactly 2^63 - 1 -> 64 bit int
    tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

    final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

    return op.Reshape(self, final_shape)

def custom_unflatten(g, self, dim, shape):
    return g.onnxscript_op(aten_unflatten, self, dim, shape).setType(self.type().with_sizes([1, 2, 3, 4, 5])) 

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::unflatten",
    symbolic_fn=custom_unflatten,
    opset_version=18,
)