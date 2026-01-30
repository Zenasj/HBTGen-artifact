import torch

with graph.inserting_after(node):
            new_node = graph.call_function(torch.ops.aten.contiguous.default, args=(node,))
            node.replace_all_uses_with(new_node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(0, node)
            node.users = {new_node: None}