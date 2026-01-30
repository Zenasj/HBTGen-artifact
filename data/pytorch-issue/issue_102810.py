import torch

class OnnxDispatcher:
    def __init__():
        self.dispatch_table = { 
            "placeholder": dispatch_placeholder,
            "call_function": dispatch_call_function,
            "output": dispatch_output,
            "call_method": dispatch_call_method,
            "call_module": dispatch_call_module,
            "get_attr": dispatch_get_attr
        }

    def dispatch_placeholder():
        pass
    def dispatch_call_function():
        pass
    def dispatch_output():
        pass
    def dispatch_call_method():
        pass
    def dispatch_call_module():
        pass
    def dispatch_get_attr():
        pass
    def dispatch(self, graph_module, onnxscript_graph, node: torch.fx.Node):
        try:
            self.dispatch_table[node.op](...)
        except KeyError:
            raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")