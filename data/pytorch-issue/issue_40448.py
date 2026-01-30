def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    if not bool(re.match(r"^[a-zA-Z0-9-_]*::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name)):
        raise RuntimeError("Failed to register operator {}. \
                           The symbolic name must match the format Domain::Name, \
                           and sould start with a letter and contain only \
                           alphanumerical characters"
                           .format(symbolic_name))
    ns, op_name = symbolic_name.split('::')
    unaccepted_domain_names = ["onnx", "aten", "prim"]
    if ns in unaccepted_domain_names:
        raise RuntimeError("Failed to register operator {}. The domain {} is already a used domain."
                           .format(symbolic_name, ns))
    import torch.onnx.symbolic_registry as sym_registry
    from torch.onnx.symbolic_helper import _onnx_stable_opsets

    for version in _onnx_stable_opsets:
        if version >= opset_version:
            sym_registry.register_op(op_name, symbolic_fn, ns, version)

raise RuntimeError("Failed to register operator {}. \
                           The symbolic name must match the format Domain::Name, \
                           and sould start with a letter and contain only \
                           alphanumerical characters"
                           .format(symbolic_name))