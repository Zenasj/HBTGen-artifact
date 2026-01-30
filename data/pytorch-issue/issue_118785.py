import torch

def is_python_constant(self):
        print(self)
        try:
            self.as_python_constant()
            return True
        except NotImplemented:
            return False

def is_python_constant(self):
        print(self)
        try:
            self.as_python_constant()
            return True
        except:
            return False

def as_python_constant(self):
        if self.original:
            return self.original
        else:

            def get_val(v):
                if isinstance(v, variables.UserDefinedObjectVariable):
                    return v.value
                else:
                    return v.as_python_constant()
            print(self.func.source)
            print(dir(self.func))
            return functools.partial(
                self.func.get_function(),
                *[get_val(arg) for arg in self.args],
                **{k: get_val(v) for k, v in self.keywords.items()},
            )

def get_function(self):
        print("here{}",functools.partial(self.func.get_function()))
        print(self.args)
        return functools.partial(self.func.get_function(), *self.args, **self.keywords)

tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    *proxy_args_kwargs(args, kwargs),
                              ),
            )

def get_fake_values_from_nodes(tx, nodes):
    def visit(n: torch.fx.Node):
        return n.meta["example_value"]

    args_kwargs = torch.fx.node.map_arg(nodes, visit)
    return tree_map_only(
        torch.Tensor, functools.partial(ensure_graph_fake, tx=tx), args_kwargs
    )

def get_fake_values_from_nodes(tx, nodes):
    def visit(n: torch.fx.Node):
        if n.op =='call_function':
            return get_fake_value(n, tx)

        return n.meta["example_value"]

    args_kwargs = torch.fx.node.map_arg(nodes, visit)
    return tree_map_only(
        torch.Tensor, functools.partial(ensure_graph_fake, tx=tx), args_kwargs
    )

def get_fake_values_from_nodes(tx, nodes):
    def visit(n: torch.fx.Node):
        ## NamedTuple can be converted to call node in create_arg, with out associating an example. 
        if n.op == "call_function" and "example_value" not in n.meta:
            return get_fake_value(n, tx)

        return n.meta["example_value"]

def build_torch_function_fn(tx, value, source):
    from .builder import SourcelessBuilder, VariableBuilder

    if not source:
        return VariableBuilder(
            tx,
            AttrSource(AttrSource(source, "__torch_function__"), "__func__"),
        )(value.__torch_function__.__func__)
    else:
        return SourcelessBuilder()(tx, value.__torch_function__.__func__)