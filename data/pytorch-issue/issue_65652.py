for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            attr_name = node.s('name')
            attr_key = str(node).split(":")[0].strip()
            parent = node.input().node()
            if parent.kind() == GETATTR_KIND:  # If the parent node is not the top-level "self" node
                parent_attr_name = parent.s('name')
                parent_attr_key = str(parent).split(":")[0].strip()
                parent_scope = attr_to_scope[parent_attr_key]
                attr_scope = parent_scope.split('/')[-1]
                attr_to_scope[attr_key] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
            else:
                attr_to_scope[attr_key] = '__module.{}'.format(attr_name)
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[attr_key]  # type: ignore
                nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))