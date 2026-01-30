import torch

def has_torch_function(
    vt: "torch._dynamo.variables.base.VariableTracker",
    tx: "torch._dynamo.symbolic_convert.InstructionTranslatorBase",
) -> bool:
    from torch._dynamo.variables import UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    if vt.has_unpack_var_sequence(tx):
        return any(has_torch_function(v, tx) for v in vt.unpack_var_sequence(tx))
    else:
        return isinstance(vt, TensorWithTFOverrideVariable) or (
            isinstance(vt, UserDefinedObjectVariable)
            and hasattr(vt.value, "__torch_function__")
        )