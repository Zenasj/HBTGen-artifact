def codegen_dynamic_scalar(self, node):
        from .cpp import DTYPE_TO_ATEN

        (data,) = (t.codegen_reference() for t in node.inputs)
        if node.is_bool:
            self.writeline(f"bool {node.sym} = {data}.item() ? 1 : 0;")
        else:
            convert_type = DTYPE_TO_ATEN[node.inputs[0].get_dtype()].replace(
                "at::k", "to"
            )
            self.writeline(f"auto {node.sym} = {data}.item().{convert_type}();")